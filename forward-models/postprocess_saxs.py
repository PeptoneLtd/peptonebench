import argparse
import functools
import logging
import os
import re
import shutil
import traceback
from multiprocessing import Pool
from typing import Union, Dict

import mdtraj as md
from pepsi_wrapper import run_pepsi
from utils import load_db, list_pdbs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_boltz_result(
    label: str,
    generator_dir: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs Pepsi-SAXS forward model on boltz1 or boltz2 predictions. This code assumes boltz was run as follows:

    while  IFS=, read -r label sequence ph ; do
        echo -n ">A|protein\n$sequence" > $label.fasta
        boltz predict $label.fasta --cache /cache --use_msa_server --out_dir generator_dir/$label \
            --diffusion_samples $num_samples --output_format pdb
    done < PeptoneDB-SAXS-sequences.csv

    :param label: label of the protein to be processed
    :param generator_dir: directory containing boltz results for all labels
    :param output_dir: directory to write results to
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
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
    saxs_data = os.path.join(saxs_data_dir, f"{label}.csv")
    output_dir = os.path.join(output_dir, f"out-{label}")
    pepsi_csv = os.path.join(pepsi_path, f"Pepsi-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isdir(output_dir) and os.path.isfile(pepsi_csv)
    ):
        try:
            combined_traj = md.load(list_pdbs(path))
            combined_traj[0].save(pdb)
            combined_traj.save_xtc(xtc)
            pH = db[label].get("pH", 7.0)
            sequence = db[label]["sequence"]
            run_pepsi(
                trajectory=xtc,
                topology=pdb,
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=pH,
                sequence=sequence,
            )
            shutil.move(os.path.join(output_dir, "saxs_curves.csv"), pepsi_csv)
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"failed to process boltz {label}: {e}")
            logger.error(traceback.format_exc())


def process_pdb_result(
    input_pdb: str,
    generator_dir: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs Pepsi-SAXS forward model on a generic pdb file which contains multiple frames

    :param input_pdb: filename of pdb file
    :param generator_dir: directory containing pdb files results for all labels
    :param output_dir: directory to write results to
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """

    path = os.path.join(generator_dir, input_pdb)
    logger.info(path)
    label = input_pdb.split(".")[0]
    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    saxs_data = os.path.join(saxs_data_dir, f"{label}.csv")
    output_dir = os.path.join(output_dir, f"out-{label}")
    pepsi_csv = os.path.join(pepsi_path, f"Pepsi-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isdir(output_dir) and os.path.isfile(pepsi_csv)
    ):
        try:
            combined_traj = md.load(path)
            combined_traj[0].save(pdb)
            combined_traj.save_xtc(xtc)
            pH = db[label].get("pH", 7.0)
            sequence = db[label]["sequence"]
            run_pepsi(
                trajectory=xtc,
                topology=pdb,
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=pH,
                sequence=sequence,
            )
            shutil.move(os.path.join(output_dir, "saxs_curves.csv"), pepsi_csv)
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"failed to process pdb {label}: {e}")
            logger.error(traceback.format_exc())


def process_esmfold_result(
    label: str,
    generator_dir: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs Pepsi-SAXS forward model on esmfold predictions. This code assumes esmfold was run as follows:

    while  IFS=, read -r label sequence ph ; do
        echo -n ">A|protein\n$sequence" > $label.fasta
        esm-fold -i $label.fasta -o generator_dir/${label}
    done < PeptoneDB-SAXS-sequences.csv

    :param label: label of the protein to be processed
    :param generator_dir: directory containing esmfold results for all labels
    :param output_dir: directory to write results to
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """

    path = os.path.join(generator_dir, f"{label}.pdb")
    if not os.path.isfile(path):
        logger.error(path)
        logger.error(f"{label} does not have valid esmfold predictions")
        return
    logger.info(path)
    pdb = os.path.join(output_dir, f"{label}.pdb")
    saxs_data = os.path.join(saxs_data_dir, f"{label}.csv")
    output_dir = os.path.join(output_dir, f"out-{label}")
    pepsi_csv = os.path.join(pepsi_path, f"Pepsi-{label}.csv")
    if overwrite or not (os.path.isfile(pdb) and os.path.isdir(output_dir) and os.path.isfile(pepsi_csv)):
        try:
            combined_traj = md.load(path)
            combined_traj[0].save(pdb)
            pH = db[label].get("pH", 7.0)
            sequence = db[label]["sequence"]
            run_pepsi(
                trajectory=pdb,
                topology=pdb,
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=pH,
                sequence=sequence,
            )
            shutil.move(os.path.join(output_dir, "saxs_curves.csv"), pepsi_csv)
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"failed to process esmfold {label}: {e}")
            logger.error(traceback.format_exc())


def process_bioemu_result(
    label: str,
    generator_dir: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs Pepsi-SAXS forward model on a pdb+xtc pair generated using bioemu sidechain reconstruction.
    This code assumes bioemu was run as follows:

    while  IFS=, read -r label sequence ph ; do
        python -m bioemu.sample --sequence "$sequence" --num_samples $num_samples --filter_samples=False \
            --output_dir generator_dir/backbone/$label/
        python -m bioemu.sidechain_relax --no-md-equil --md-protocol md_equil \
            --pdb-path generator_dir/backbone/$label/topology.pdb \
            --xtc-path generator_dir/backbone/$label/samples.xtc --outpath generator_dir/sidechains/$label
    done < PeptoneDB-SAXS-sequences.csv


    :param label: label of the protein
    :param generator_dir: directory containing bioemu results for all labels
    :param output_dir: directory to write results to
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """
    path = os.path.join(generator_dir, f"{label}/samples_sidechain_rec.pdb")
    xtc_path = os.path.join(generator_dir, f"{label}/samples_sidechain_rec.xtc")
    if not os.path.isfile(path):
        logger.error(path)
        logger.error(f"{label} does not have valid esmfold predictions")
        return
    logger.info(path)
    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    saxs_data = os.path.join(saxs_data_dir, f"{label}.csv")
    output_dir = os.path.join(output_dir, f"out-{label}")
    pepsi_csv = os.path.join(pepsi_path, f"Pepsi-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isdir(output_dir) and os.path.isfile(pepsi_csv)
    ):
        try:
            combined_traj = md.load(xtc_path, top=path)
            combined_traj[0].save(pdb)
            combined_traj.save(xtc)
            pH = db[label].get("pH", 7.0)
            sequence = db[label]["sequence"]
            run_pepsi(
                trajectory=xtc,
                topology=pdb,
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=pH,
                sequence=sequence,
            )
            shutil.move(os.path.join(output_dir, "saxs_curves.csv"), pepsi_csv)
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"failed to process bioemu {label}: {e}")
            logger.error(traceback.format_exc())


def process_alphafold_result(
    pdb_file: str,
    generator_dir: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs Pepsi-SAXS forward model on alphafold2 prediction. We only evaluate the 1st ranked aplhafold prediction.
    This code assumes alphafold2 was ran using the public colabfold available at
    https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb and that
    generator_dir contains the unzipped results:

    :param pdb_file: filename of the pdbfile to be processed
    :param generator_dir: directory containing alphafold2 results
    :param output_dir: directory to write results to
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
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
    saxs_data = os.path.join(saxs_data_dir, f"{label}.csv")
    output_dir = os.path.join(output_dir, f"out-{label}")
    pepsi_csv = os.path.join(pepsi_path, f"Pepsi-{label}.csv")
    if overwrite or not (os.path.isfile(pdb) and os.path.isdir(output_dir) and os.path.isfile(pepsi_csv)):
        try:
            combined_traj = md.load(path)
            combined_traj[0].save(pdb)
            pH = db[label].get("pH", 7.0)
            sequence = db[label]["sequence"]
            run_pepsi(
                trajectory=pdb,
                topology=pdb,
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=pH,
                sequence=sequence,
            )
            shutil.move(os.path.join(output_dir, "saxs_curves.csv"), pepsi_csv)
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"failed to process alphafold {label}: {e}")
            logger.error(traceback.format_exc())


def process_idpsam_result(
    label: str,
    generator_dir: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    overwrite: bool = False,
) -> None:
    """
    Runs Pepsi-SAXS forward model on idpsam prediction. This code assumes idpsam was run as follows:

    while  IFS=, read -r label sequence ph ; do
       generate_ensemble.py -c config/models.yaml -s $seq -o generator_dir/$label/$label -b 25 -n 1000 -a -d cuda
    done < PeptoneDB-SAXS-sequences.csv

    :param label: label of the protein
    :param generator_dir: directory containing bioemu results for all labels
    :param output_dir: directory to write results to
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
    :param db: dictionary containing sequences and pH levels for each label
    :param overwrite: overwrite existing output files
    """

    top_path = os.path.join(generator_dir, label, f"{label}.aa.top.pdb")
    traj_path = os.path.join(generator_dir, label, f"{label}.aa.traj.dcd")

    pdb = os.path.join(output_dir, f"{label}.pdb")
    xtc = os.path.join(output_dir, f"{label}.xtc")
    saxs_data = os.path.join(saxs_data_dir, f"{label}.csv")
    if not os.path.isfile(top_path):
        logger.error(f"{top_path} missing")
        return
    if not os.path.isfile(traj_path):
        logger.error(f"{traj_path} missing")
        return

    output_dir = os.path.join(output_dir, f"out-{label}")
    pepsi_csv = os.path.join(pepsi_path, f"Pepsi-{label}.csv")
    if overwrite or not (
        os.path.isfile(pdb) and os.path.isfile(xtc) and os.path.isdir(output_dir) and os.path.isfile(pepsi_csv)
    ):
        try:
            combined_traj = md.load(traj_path, top=top_path)
            combined_traj[0].save(pdb)
            combined_traj.save(xtc)
            pH = db[label].get("pH", 7.0)
            sequence = db[label]["sequence"]
            run_pepsi(
                trajectory=xtc,
                topology=pdb,
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=pH,
                sequence=sequence,
            )
            shutil.move(os.path.join(output_dir, "saxs_curves.csv"), pepsi_csv)
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"failed to process idpsam {label}: {e}")
            logger.error(traceback.format_exc())


def process(
    generators_dir: str,
    generator_name: str,
    output_dir: str,
    saxs_data_dir: str,
    pepsi_path: str,
    db: Dict[str, Dict[str, Union[str, float]]],
    nproc: int = 8,
) -> None:
    """
    Parallely (nproc processes) runs the Pepsi-SAXS forward model on predictions from a single ensmble generator

    :param generators_dir: directory containing one folder for each generator
    :param generator_name: name of the generator to be processed
    :param output_dir: results are written to output_dir/generator_name
    :param saxs_data_dir: directory containing saxs data
    :param pepsi_path: path to Pepsi-SAXS executable
    :param db: dictionary containing sequences and pH levels for each label
    :param nproc: how many processes to use
    """
    generator_dir = generator_name
    f = process_pdb_result
    if generator_name == "bioemu":
        generator_dir = "bioemu/sidechains"
        f = process_bioemu_result
    elif generator_name == "idpfold":
        generator_dir = "idpfold/aa"
        f = process_bioemu_result
    elif generator_name == "idpgan":
        generator_dir = "idpgan/aa"
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

    generator_dir = os.path.join(generators_dir, generator_dir)
    os.makedirs(output_dir, exist_ok=True)

    f = functools.partial(
        f, generator_dir=generator_dir, output_dir=output_dir, saxs_data_dir=saxs_data_dir, pepsi_path=pepsi_path, db=db
    )
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

    parser.add_argument(
        "--saxs-data",
        action="store",
        dest="saxs_data",
        help="path to saxs data",
        type=str,
        default="/workdir/saxs_database",
    )

    parser.add_argument(
        "--pepsi",
        action="store",
        dest="pepsi",
        help="path to pepsi executable",
        type=str,
        default="/home/mambauser/Pepsi-SAXS",
    )

    parser.add_argument(
        "--dataset",
        action="store",
        dest="dataset",
        help="path dataset csv file",
        type=str,
        default="PeptoneDB-SAXS-sequences.csv",
    )

    return parser.parse_args()


def main():
    try:
        args = get_args()
    except SystemExit:
        raise ValueError("could not parse command-line arguments")

    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)

    db = load_db(args.dataset)

    for generator in args.generators.split(","):
        output_dir = str(os.path.join(args.workdir, "processed-saxs", generator))
        process(
            generators_dir=args.workdir,
            generator_name=generator,
            output_dir=output_dir,
            saxs_data_dir=args.saxs_data,
            pepsi_path=args.pepsi,
            nproc=args.nprocs,
            db=db,
        )


if __name__ == "__main__":
    main()
