import argparse
import functools
import logging
import os
import re
import traceback
from multiprocessing import Pool
from typing import Dict, Union

import CSpred as UCBshift
import mdtraj as md
import pandas as pd
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from .utils import list_pdbs, load_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def addh_chemshifts(
    trj: md.Trajectory,
    label: str,
    outpath: str,
    pH: float = 7.0,
    tmpdir: str = "/tmp",
) -> None:
    """
    Predict chemical shifts for traj using Sparta+ and UCBShift.
    Predictions are made frame by frame, and concatenated.
    Each frame is passed through PDBFixer to set the correct protonation state for given pH prior to being used as
    input for CS predictors
    :param trj: mdtraj trajectory
    :param label: label of the protein
    :param outpath: where to store the predictions
    :param pH: pH, defaults to 7.0
    :param tmpdir: directory for temporary files
    """
    spartap_df = None
    UCBshift_df = None
    spartap_csv = os.path.join(outpath, f"Sparta+-{label}.csv")
    UCBshift_csv = os.path.join(outpath, f"UCBshift-{label}.csv")

    for i, frame in enumerate(trj):
        filename_i = os.path.join(tmpdir, f"{label}_{i}.pdb")
        addh_filename_i = os.path.join(tmpdir, f"{label}_{i}-addH.pdb")
        trj[i].save(filename_i)

        fixer = PDBFixer(filename=filename_i)
        fixer.addMissingHydrogens(pH)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(addh_filename_i, "w"))
        tmp_trj = md.load(addh_filename_i)
        if not os.path.isfile(spartap_csv):
            if spartap_df is None:
                spartap_df = md.chemical_shifts_spartaplus(tmp_trj)
            else:
                spartap_df = pd.concat(
                    [spartap_df, md.chemical_shifts_spartaplus(tmp_trj).rename(columns={0: i})], axis=1
                )
        if not os.path.isfile(UCBshift_csv):
            df = UCBshift.calc_sing_pdb(addh_filename_i, pH, TP=False, ML=True, test=False)
            df["frame"] = i
            df = df.set_index(["RESNUM", "RESNAME", "frame"]).stack()
            df.index.names = ["resSeq", None, "frame", "name"]
            df = pd.pivot_table(df.to_frame(name="x"), values="x", index="frame", columns=["resSeq", "name"])
            df.columns = pd.MultiIndex.from_tuples([(str(a[0]), a[1][:-2]) for a in df.columns], names=df.columns.names)
            df = df.reset_index().drop(columns="frame").T
            if UCBshift_df is None:
                UCBshift_df = df
            else:
                UCBshift_df = pd.concat([UCBshift_df, df.rename(columns={0: i})], axis=1)

        os.remove(addh_filename_i)
        os.remove(filename_i)
    if not os.path.isfile(spartap_csv):
        spartap_df.to_csv(spartap_csv)
    if not os.path.isfile(UCBshift_csv):
        UCBshift_df.to_csv(UCBshift_csv)


def process_boltz_result(
    label: str,
    generator_dir: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
        Run CS forward models for boltz1 or boltz2 predictions. This code assumes boltz was run as follows:

        while  IFS=, read -r label sequence ph gscores ; do
            echo -n ">A|protein\n$sequence" > $label.fasta
            boltz predict $label.fasta --cache /cache --use_msa_server --out_dir generator_dir/$label \
                --diffusion_samples $num_samples --output_format pdb
        done < PeptoneDB-SAXS-sequences.csv

        :param label: label of the protein to be processed
        :param generator_dir: directory containing boltz results for all labels
        :param output_dir: directory to write results to
        :param db: dictionary containing sequences and pH levels for each label
        :param overwrite: overwrite existing output files
        """
    path = os.path.join(generator_dir, f"{label}/boltz_results_{label}/predictions/{label}")
    if not os.path.isdir(path):
        logger.error(f"{label} does not have valid boltz predictions")
        return
    logger.info(path)
    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    spartap = os.path.join(output_dir, f"Sparta+-{label}.csv")
    UCBshift = os.path.join(output_dir, f"UCBshift-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isfile(spartap) and os.path.isfile(UCBshift)
    ):
        try:
            combined_traj = md.load(list_pdbs(path))
            combined_traj[0].save(pdb)
            combined_traj.save_xtc(xtc)
            pH = db[label].get("pH", 7.0)
            addh_chemshifts(combined_traj, label, output_dir, pH=pH)
        except Exception as e:
            logger.error(f"failed to process chemshifts for boltz {label}: {e}")
            logger.error(traceback.format_exc())


def process_pdb_result(
    input_pdb: str,
    generator_dir: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs CS forward models on a generic pdb file which contains multiple frames

    :param input_pdb: filename of pdb file
    :param generator_dir: directory containing pdb files results for all labels
    :param output_dir: directory to write results to
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """
    path = os.path.join(generator_dir, input_pdb)
    logger.info(path)
    label = input_pdb.split(".")[0]
    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    spartap = os.path.join(output_dir, f"Sparta+-{label}.csv")
    UCBshift = os.path.join(output_dir, f"UCBshift-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isfile(spartap) and os.path.isfile(UCBshift)
    ):
        try:
            combined_traj = md.load(path)
            combined_traj[0].save(pdb)
            combined_traj.save_xtc(xtc)
            pH = db[label].get("pH", 7.0)
            addh_chemshifts(combined_traj, label, output_dir, pH=pH)
        except Exception as e:
            logger.error(f"failed to process chemshifts for pdb {label}: {e}")
            logger.error(traceback.format_exc())


def process_esmfold_result(
    pdb_dir: str,
    generator_dir: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs CS forward models on esmfold predictions. This code assumes esmfold was run as follows:

    while  IFS=, read -r label sequence ph gscores ; do
        echo -n ">A|protein\n$sequence" > $label.fasta
        esm-fold -i $label.fasta -o generator_dir/${label}
    done < PeptoneDB-SAXS-sequences.csv

    :param label: label of the protein to be processed
    :param generator_dir: directory containing esmfold results for all labels
    :param output_dir: directory to write results to
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """
    label = pdb_dir.split(".")[0]
    path = os.path.join(generator_dir, f"{pdb_dir}/A|protein.pdb")
    if not os.path.isfile(path):
        logger.error(path)
        logger.error(f"{label} does not have valid esmfold predictions")
        return
    logger.info(path)
    pdb = os.path.join(output_dir, f"{label}.pdb")
    spartap = os.path.join(output_dir, f"Sparta+-{label}.csv")
    UCBshift = os.path.join(output_dir, f"UCBshift-{label}.csv")
    if overwrite or not (os.path.isfile(pdb) and os.path.isfile(spartap) and os.path.isfile(UCBshift)):
        try:
            combined_traj = md.load(path)
            combined_traj[0].save(pdb)
            pH = db[label].get("pH", 7.0)
            addh_chemshifts(combined_traj, label, output_dir, pH=pH)
        except Exception as e:
            logger.error(f"failed to process chemshifts for esmfold {label}: {e}")
            logger.error(traceback.format_exc())


def process_bioemu_result(
    label: str,
    generator_dir: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs CS forward models on a pdb+xtc pair generated using bioemu sidechain reconstruction.
    This code assumes bioemu was run as follows:

    while  IFS=, read -r label sequence ph gscores ; do
        python -m bioemu.sample --sequence "$sequence" --num_samples $num_samples --filter_samples=False \
            --output_dir generator_dir/backbone/$label/
        python -m bioemu.sidechain_relax --no-md-equil --md-protocol md_equil \
            --pdb-path generator_dir/backbone/$label/topology.pdb \
            --xtc-path generator_dir/backbone/$label/samples.xtc --outpath generator_dir/sidechains/$label
    done < PeptoneDB-SAXS-sequences.csv


    :param label: label of the protein
    :param generator_dir: directory containing bioemu results for all labels
    :param output_dir: directory to write results to
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """
    path = os.path.join(generator_dir, f"{label}/samples_sidechain_rec.pdb")
    xtc_path = os.path.join(generator_dir, f"{label}/samples_sidechain_rec.xtc")
    if not os.path.isfile(path):
        logger.error(path)
        logger.error(f"{label} does not have valid bioemu predictions")
        return
    logger.info(path)
    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    spartap = os.path.join(output_dir, f"Sparta+-{label}.csv")
    UCBshift = os.path.join(output_dir, f"UCBshift-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isfile(spartap) and os.path.isfile(UCBshift)
    ):
        try:
            combined_traj = md.load(xtc_path, top=path)
            combined_traj[0].save(pdb)
            combined_traj.save(xtc)
            pH = db[label].get("pH", 7.0)
            addh_chemshifts(combined_traj, label, output_dir, pH=pH)
        except Exception as e:
            logger.error(f"failed to process chemshifts for bioemu {label}: {e}")
            logger.error(traceback.format_exc())


def process_alphafold_result(
    pdb_file: str,
    generator_dir: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs CS forward models on alphafold2 prediction. We only evaluate the 1st ranked aplhafold prediction.
    This code assumes alphafold2 was ran using the public colabfold available at
    https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb and that
    generator_dir contains the unzipped results:

    :param pdb_file: filename of the pdbfile to be processed
    :param generator_dir: directory containing alphafold2 results
    :param output_dir: directory to write results to
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """
    m = re.match(r"(.*)_unrelaxed_rank_001_alphafold2_ptm_model_.*_seed_.*\.pdb", pdb_file)
    if not m:
        return
    label = m.group(1)
    path = os.path.join(generator_dir, pdb_file)
    if not os.path.isfile(path):
        logger.error(path)
        logger.error(f"{label} does not have valid esmfold predictions")
        return
    logger.info(path)
    pdb = os.path.join(output_dir, f"{label}.pdb")
    spartap = os.path.join(output_dir, f"Sparta+-{label}.csv")
    UCBshift = os.path.join(output_dir, f"UCBshift-{label}.csv")
    if overwrite or not (os.path.isfile(pdb) and os.path.isfile(spartap) and os.path.isfile(UCBshift)):
        try:
            combined_traj = md.load(path)
            combined_traj[0].save(pdb)
            pH = db[label].get("pH", 7.0)
            addh_chemshifts(combined_traj, label, output_dir, pH=pH)
        except Exception as e:
            logger.error(f"failed to process chemshifts for alphafold {label}: {e}")
            logger.error(traceback.format_exc())


def process_idpsam_result(
    label: str,
    results_dir: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs CS forward models on idpsam prediction. This code assumes idpsam was run as follows:

    while  IFS=, read -r label sequence ph gscores ; do
       generate_ensemble.py -c config/models.yaml -s $seq -o generator_dir/$label/$label -b 25 -n 1000 -a -d cuda
    done < PeptoneDB-SAXS-sequences.csv

    :param label: label of the protein
    :param generator_dir: directory containing bioemu results for all labels
    :param output_dir: directory to write results to
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """

    top_path = os.path.join(results_dir, label, f"{label}.aa.top.pdb")
    traj_path = os.path.join(results_dir, label, f"{label}.aa.traj.dcd")

    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    spartap = os.path.join(output_dir, f"Sparta+-{label}.csv")
    UCBshift = os.path.join(output_dir, f"UCBshift-{label}.csv")

    if not os.path.isfile(top_path):
        logger.error(f"{top_path} missing")
        return
    if not os.path.isfile(traj_path):
        logger.error(f"{traj_path} missing")
        return

    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isfile(spartap) and os.path.isfile(UCBshift)
    ):
        try:
            traj = md.load(traj_path, top=top_path)
            traj[0].save(pdb)
            traj.save(xtc)
            pH = db[label].get("pH", 7.0)
            addh_chemshifts(traj, label, output_dir, pH=pH)
        except Exception as e:
            logger.error(f"failed to process chemshifts for idpsam {label}: {e}")
            logger.error(traceback.format_exc())


def process(
    generators_dir: str,
    generator_name: str,
    output_dir: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    nproc: int = 8,
) -> None:
    """
    Parallely (nproc processes) runs the Sparta+ and UCBShift forward models on predictions from a single
    ensemble generator

    :param generators_dir: directory containing one folder for each generator
    :param generator_name: name of the generator to be processed
    :param output_dir: results are written to output_dir/generator_name
    :param db: dictionary containing sequences and pH levels for each label
    :param nproc: how many processes to use
    """
    preds_dir = generator_name
    f = process_pdb_result
    if generator_name == "bioemu":
        preds_dir = "bioemu/sidechains"
        f = process_bioemu_result
    elif generator_name == "idpfold":
        preds_dir = "idpfold/aa"
        f = process_bioemu_result
    elif generator_name == "idpgan":
        preds_dir = "idpgan/aa"
        f = process_pdb_result
    elif generator_name.startswith("boltz"):
        f = process_boltz_result
    elif generator_name == "esmfold":
        f = process_esmfold_result
    elif generator_name == "alphafold":
        f = process_alphafold_result
    elif generator_name.startswith("peptron"):
        f = process_pdb_result
    elif generator_name == "idpsam":
        f = process_idpsam_result

    generator_dir = os.path.join(generators_dir, preds_dir)
    os.makedirs(output_dir, exist_ok=True)

    f = functools.partial(f, generator_dir=generator_dir, output_dir=output_dir, db=db)
    pool = Pool(nproc)
    pool.map(f, os.listdir(generator_dir))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        action="store",
        dest="workdir",
        help="local work directory",
        default="/workdir",
    )
    parser.add_argument(
        "--generators",
        action="store",
        dest="generators",
        help="generators to post-process",
        default="bioemu,boltz1x,esmflow,esmfold,idpgan,idp-o",
    )
    parser.add_argument(
        "--nprocs",
        action="store",
        dest="nprocs",
        help="number of processors to use",
        type=int,
        default=8,
    )

    parser.add_argument("--dataset", action="store", dest="dataset", help="path to dataset csv", type=str)
    return parser.parse_args()


def main():
    try:
        args = get_args()
    except SystemExit:
        raise ValueError("could not parse command-line arguments")

    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)

    db = load_db(args.dataset)

    for pred in args.generators.split(","):
        processed_dir = str(os.path.join(args.workdir, "processed", pred))
        process(args.workdir, processed_dir, pred, db=db, nproc=args.nprocs)


if __name__ == "__main__":
    main()
