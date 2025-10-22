# Copyright 2025 Peptone Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import shutil
from glob import glob

import joblib
import mdtraj as md
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from cspred_wrapper import run_cspred
from pepsi_wrapper import run_pepsi

N_JOBS = min(joblib.cpu_count(), int(os.getenv("N_JOBS", "64")))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forward model calculations on protein ensembles."
        "Set number of processes with N_JOBS env variable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--predictor",
        type=str,
        choices=["Pepsi", "UCBshift", "Sparta+", "UCBshift_Sparta+"],
        required=True,
        help="Forward-model predictor to use",
    )
    parser.add_argument(
        "-f",
        "--generator-dir",
        type=str,
        nargs="+",
        default=["."],
        help="folder(s) with generated protein ensembles to be analyzed and reweighted",
    )
    parser.add_argument(
        "--output-dir",
        help="directory where to store results. Default is to use the same directory as the generator",
        default=None,
    )
    parser.add_argument(
        "--prepare-ensembles",
        action="store_true",
        help="Save ensembles into XTC+PDB before running forward models",
    )
    parser.add_argument(
        "--no-prepare-ensembles",
        action="store_false",
        dest="prepare_ensembles",
        help="Do not prepare ensembles, they are already in XTC+PDB format",
    )
    parser.set_defaults(prepare_ensembles=False)
    parser.add_argument(
        "--dataset",
        help="path to PeptoneDB csv, leave empty to use default CS and SAXS locations",
        type=str,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing forward model prediction files")

    # SAXS-related arguments
    parser.add_argument(
        "--saxs-data",
        help="path to SAXS experimental data, needed to run Pepsi",
        type=str,
        default="/app/peptonebench/datasets/PeptoneDB-SAXS/sasbdb-clean_data",
    )
    parser.add_argument(
        "--saxs-filename",
        help="filename inside saxs_data folder. "
        "PeptoneBD-CS uses 'LABEL-bift.dat', while PeptoneDB-Integrative uses 'LABEL/SAXS_bift.dat'",
        type=str,
        default="LABEL-bift.dat",
    )
    parser.add_argument(
        "--pepsi-path",
        help="path to Pepsi-SAXS executable",
        type=str,
        default="/home/mambauser/Pepsi-SAXS",
    )
    return parser.parse_args()


def save_as_xtc_and_pdb(filename: str, generator_dir: str, output_dir: str, label: str = None) -> None:
    """Load a trajectory file (PDB or HDF5) and save first frame as PDB and entire trajectory as XTC."""
    if not (filename.endswith((".pdb", ".h5"))):
        raise ValueError(f"Unsupported file format: {filename}")
    if label is None:
        label = os.path.basename(filename).split(".")[0]
    trj = md.load(os.path.join(generator_dir, filename))
    trj[0].save(os.path.join(output_dir, label + ".pdb"))
    trj.save(os.path.join(output_dir, label + ".xtc"))


def prepare_ensembles_custom(generator_dir: str, output_dir: str, generator_name: str = None) -> None:
    """Prepare ensemble files from different generators saving them into XTC + PDB format.
    Supported generators: 
    - 'alphafold': We only evaluate the 1st ranked alphafold prediction. 
        This code assumes alphafold2 was ran using the public colabfold available at
        https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb
    - 'bioemu': This code assumes bioemu was run as follows:
        while  IFS=, read -r label sequence ph gscores ; do
            python -m bioemu.sample --sequence "$sequence" --num_samples $num_samples --filter_samples=False \
                --output_dir generator_dir/backbone/$label/
            python -m bioemu.sidechain_relax --no-md-equil --md-protocol md_equil \
                --pdb-path generator_dir/backbone/$label/topology.pdb \
                --xtc-path generator_dir/backbone/$label/samples.xtc --outpath generator_dir/sidechains/$label
        done < PeptoneDB.csv
    - 'boltz2' and 'boltz1x': This code assumes boltz was run as follows:
        while  IFS=, read -r label sequence ph gscores ; do
            echo -n ">A|protein\n$sequence" > $label.fasta
            boltz predict $label.fasta --cache /cache --use_msa_server --out_dir generator_dir/$label \
                --diffusion_samples $num_samples --output_format pdb
        done < PeptoneDB.csv
    - 'esmflow': trajectories in LABEL.pdb format
    - 'esmfold': This code assumes esmfold was run as follows:
        while  IFS=, read -r label sequence ph gscores ; do
            echo -n ">A|protein\n$sequence" > $label.fasta
            esm-fold -i $label.fasta -o generator_dir/${label}
        done < PeptoneDB.csv
    - 'idp-o': trajectories in LABEL.pdb or LABEL.h5 format
    - 'idpfold': same sidechain reconstruction as bioemu
    - 'idpgan': cg2all reconstruction
    - 'idpsam': This code assumes idpsam was run as follows:
        while  IFS=, read -r label sequence ph gscores ; do
        generate_ensemble.py -c config/models.yaml -s $seq -o generator_dir/$label/$label -b 25 -n 1000 -a -d cuda
        done < PeptoneDB.csv
    - 'peptron': trajectories in LABEL.pdb format
    """
    if generator_name is None:
        generator_name = os.path.basename(generator_dir)
    if generator_name == "alphafold":
        for pdb_file in sorted(
            glob(f"{generator_dir}/*_unrelaxed_rank_001_alphafold2_ptm_model_*.pdb"),
            key=lambda x: (len(x), x),
        ):
            label = os.path.basename(pdb_file).split("_unrelaxed_")[0]
            save_as_xtc_and_pdb(pdb_file, generator_dir, output_dir, label)
    elif generator_name == "bioemu":
        for file in sorted(glob(f"{generator_dir}/*/samples_sidechain_rec.xtc"), key=lambda x: (len(x), x)):
            if os.path.exists(file.replace(".xtc", ".pdb")):
                label = os.path.basename(os.path.dirname(file))
                shutil.copy(file, os.path.join(output_dir, label + ".xtc"))
                shutil.copy(file.replace(".xtc", ".pdb"), os.path.join(output_dir, label + ".pdb"))
    elif generator_name == "boltz2" or generator_name == "boltz1x":
        for label in sorted(os.listdir(generator_dir), key=lambda x: (len(x), x)):
            pattern = f"{generator_dir}/{label}/boltz_results_{label}/predictions/{label}/{label}_model_*.pdb"
            list_pdbs = sorted(glob(pattern), key=lambda x: (len(x), x))
            if list_pdbs:
                trj = md.load(list_pdbs)
                trj[0].save(os.path.join(output_dir, label + ".pdb"))
                trj.save(os.path.join(output_dir, label + ".xtc"))
    elif generator_name == "esmfold":
        for pdb_file in sorted(glob(os.path.join(generator_dir, "*", "*.pdb")), key=lambda x: (len(x), x)):
            save_as_xtc_and_pdb(pdb_file, os.path.dirname(pdb_file), output_dir)
    elif generator_name == "idpfold":
        for file in sorted(glob(f"{generator_dir}/aa/*/samples_sidechain_rec.xtc"), key=lambda x: (len(x), x)):
            if os.path.exists(file.replace(".xtc", ".pdb")):
                label = os.path.basename(os.path.dirname(file))
                shutil.copy(file, os.path.join(output_dir, label + ".xtc"))
                shutil.copy(file.replace(".xtc", ".pdb"), os.path.join(output_dir, label + ".pdb"))
    elif generator_name == "idpgan":
        for pdb_file in sorted(glob(os.path.join(generator_dir, "aa", "*.pdb")), key=lambda x: (len(x), x)):
            save_as_xtc_and_pdb(pdb_file, os.path.dirname(pdb_file), output_dir)
    elif generator_name == "idpsam":
        for file in sorted(glob(os.path.join(generator_dir, "*.aa.traj.dcd")), key=lambda x: (len(x), x)):
            if os.path.exists(file.replace("traj.dcd", "top.pdb")):
                label = os.path.basename(file).split(".")[0]
                trj = md.load(file, top=file.replace("traj.dcd", "top.pdb"))
                trj[0].save(os.path.join(output_dir, label + ".pdb"))
                trj.save(os.path.join(output_dir, label + ".xtc"))
    else:  # esmflow, idp-o, peptron
        for ext in ["pdb", "h5"]:
            for file in sorted(glob(os.path.join(generator_dir, f"*.{ext}")), key=lambda x: (len(x), x)):
                save_as_xtc_and_pdb(file, os.path.dirname(file), output_dir)


def process_generator(
    predictor: str,
    generator_dir: str,
    output_dir: str,
    db: pd.DataFrame,
    prepare_ensembles: bool = True,
    overwrite: bool = False,
    saxs_data: str = "",
    saxs_filename: str = "LABEL-bift.dat",
    pepsi_path: str = "",
) -> None:
    """
    Runs the selected predictor forward model on ensembles from a single protein generator directory.

    Ensembles are optionally prepared (converted to XTC+PDB format) and processed in parallel using N_JOBS.
    Results are written to the specified output directory.

    :param predictor: Forward-model predictor to use ("Pepsi", "UCBshift", "Sparta+", "UCBshift_Sparta+")
    :param generator_dir: Directory containing generated protein ensembles
    :param output_dir: Directory to store results
    :param db: Pandas DataFrame containing sequences and pH levels for each label
    :param prepare_ensembles: Whether to prepare ensembles into XTC+PDB format
    :param overwrite: Overwrite existing prediction files
    :param saxs_data: Path to SAXS experimental data (for Pepsi)
    :param saxs_filename: Filename pattern for SAXS data (for Pepsi)
    :param pepsi_path: Path to Pepsi-SAXS executable
    """
    logger.info(f"--- processing '{generator_dir}' with '{predictor}' ---")
    os.makedirs(output_dir, exist_ok=True)
    if prepare_ensembles:
        logger.info(
            f"prepare_ensembles=True: copying ensembles from '{generator_dir}' to '{output_dir}' in XTC+PDB format",
        )
        prepare_ensembles_custom(generator_dir, output_dir)
        generator_dir = output_dir
    else:
        logger.info(f"prepare_ensembles=False: expecting XTC+PDB ensembles in '{generator_dir}'")

    if predictor == "Pepsi":

        def process_ensemble(label: str) -> None:
            return run_pepsi(
                trajectory=os.path.join(generator_dir, f"{label}.xtc"),
                topology=os.path.join(generator_dir, f"{label}.pdb"),
                saxs=os.path.join(saxs_data, saxs_filename.replace("LABEL", label)),
                output=output_dir,
                pepsi=pepsi_path,
                pH=db.loc[label, "pH"],
                sequence=db.loc[label, "sequence"],
                angular_units=1,  # correct units for bift processed data in PeptoneDB-SAXS and PeptoneDB-Integrative
                overwrite=overwrite,
            )
    elif predictor in ("UCBshift", "Sparta+", "UCBshift_Sparta+"):

        def process_ensemble(label: str) -> None:
            return run_cspred(
                trajectory=os.path.join(generator_dir, f"{label}.xtc"),
                topology=os.path.join(generator_dir, f"{label}.pdb"),
                output=output_dir,
                predictor=predictor.split("_"),
                pH=db.loc[label, "pH"],
                sequence=db.loc[label, "sequence"],
                overwrite=overwrite,
            )
    else:
        raise ValueError(f"predictor {predictor} not supported")

    labels = sorted([os.path.basename(file).replace(".xtc", "") for file in glob(os.path.join(generator_dir, "*.xtc"))])
    logger.info(f"found {len(labels):_} ensembles to process")
    if set(labels) - set(db.index):
        raise ValueError(
            f"some labels in '{generator_dir}' are not in the provided dataset: {set(labels) - set(db.index)}",
        )
    joblib.Parallel(n_jobs=min(N_JOBS, len(labels)))(joblib.delayed(process_ensemble)(label) for label in labels)


def main() -> None:
    args = get_args()
    if args.dataset is None:
        if args.predictor == "Pepsi":
            args.dataset = "/app/peptonebench/datasets/PeptoneDB-SAXS/PeptoneDB-SAXS.csv"
        elif args.predictor in ("UCBshift", "Sparta+", "UCBshift_Sparta+"):
            args.dataset = "/app/peptonebench/datasets/PeptoneDB-CS/PeptoneDB-CS.csv"
        else:
            raise ValueError(f"predictor {args.predictor} not supported")
    db = pd.read_csv(args.dataset, index_col="label")
    logger.info(f"loaded dataset from {args.dataset} with {len(db)} entries")
    logger.info(f"processing directories: {args.generator_dir}")
    if args.overwrite:
        logger.info("overwrite=True: existing forward model prediction files will be recomputed")
    else:
        logger.info("overwrite=False: existing forward model prediction files will be skipped")
    if args.prepare_ensembles and args.output_dir is None:
        default_output_dir = "processed_ensembles"
        logger.info(f"'prepare_ensembles=True': setting --output-dir to '{default_output_dir}' to avoid overwriting")
        args.output_dir = default_output_dir
    for generator_dir in args.generator_dir:
        if args.output_dir is None:
            processed_dir = generator_dir
        else:
            processed_dir = os.path.join(args.output_dir, os.path.basename(generator_dir))
        os.makedirs(processed_dir, exist_ok=True)
        process_generator(
            predictor=args.predictor,
            generator_dir=generator_dir,
            output_dir=processed_dir,
            db=db,
            prepare_ensembles=args.prepare_ensembles,
            overwrite=args.overwrite,
            saxs_data=args.saxs_data,
            saxs_filename=args.saxs_filename,
            pepsi_path=args.pepsi_path,
        )


if __name__ == "__main__":
    main()
