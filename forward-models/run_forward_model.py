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
from multiprocessing import Pool

import mdtraj as md
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from cspred_wrapper import run_cspred
from pepsi_wrapper import run_pepsi


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forward model calculations on ensembles generated with different generators."
        " If ensembles are already in XTC+PDB format, use --no-prepare-ensembles to skip conversion step, "
        "and set --output-dir to the same directory containing the ensembles (--generators-dir).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictor",
        type=str,
        choices=["Pepsi", "UCBshift", "Sparta+", "UCBshift_Sparta+"],
        required=True,
        help="Forward-model predictor to use",
    )
    parser.add_argument(
        "--generators-dir",
        help="directory containing one folder for each generator",
        default="./generators",
    )
    parser.add_argument(
        "--generators",
        help="generators to post-process, comma separated",
        default="",
    )
    parser.add_argument(
        "--dataset",
        help="path to PeptoneDB csv, leave empty to use default CS and SAXS locations",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="directory where to store results",
        default="./processed",
    )
    parser.add_argument(
        "--nprocs",
        help="number of processors to use",
        type=int,
        default=8,
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
        help="Do not prepare ensembles",
    )
    parser.set_defaults(prepare_ensembles=True)

    # SAXS-related arguments
    parser.add_argument(
        "--saxs-data",
        help="path to SAXS experimental data",
        type=str,
        default="/app/peptonebench/datasets/PeptoneDB-SAXS/sasbdb-clean_data",
    )
    parser.add_argument(
        "--pepsi-path",
        help="path to pepsi executable",
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


def prepare_ensembles_custom(generator_name: str, generator_dir: str, output_dir: str) -> None:
    """Prepare ensemble files from different generators saving them into XTC + PDB format.
    Supported generators: 
    - 'alphafold': We only evaluate the 1st ranked alphafold prediction. This code assumes alphafold2 was ran using the public colabfold available at
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
    generators_dir: str,
    generator_name: str,
    output_dir: str,
    db: pd.DataFrame,
    prepare_ensembles: bool = True,
    nproc: int = 8,
    saxs_data: str = "",
    pepsi_path: str = "",
) -> None:
    """
    Parallely (nproc processes) runs the predictor forward models on ensembles from a single
    protein generator

    :param predictor: which forward-model predictor to use
    :param generators_dir: directory containing one folder for each generator
    :param generator_name: name of the generator to be processed
    :param output_dir: results are written to output_dir/generator_name
    :param db: dictionary containing sequences and pH levels for each label
    :param nproc: how many processes to use
    """
    logger.info(f"processing {generator_name} with {predictor}")
    generator_dir = os.path.join(generators_dir, generator_name)
    os.makedirs(output_dir, exist_ok=True)
    if prepare_ensembles:
        prepare_ensembles_custom(generator_name, generator_dir, output_dir)

    if predictor == "Pepsi":

        def process_ensemble(label: str) -> None:
            return run_pepsi(
                trajectory=os.path.join(generator_dir, f"{label}.xtc"),
                topology=os.path.join(generator_dir, f"{label}.pdb"),
                saxs=saxs_data,
                output=output_dir,
                pepsi=pepsi_path,
                pH=db.loc[label, "pH"],
                sequence=db.loc[label, "sequence"],
                angular_units=1,
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
            )
    else:
        raise ValueError(f"predictor {predictor} not supported")

    pool = Pool(nproc)
    pool.map(process_ensemble, sorted(os.listdir(generator_dir)))


def main() -> None:
    try:
        args = get_args()
    except SystemExit:
        raise ValueError("could not parse command-line arguments")

    if (not args.prepare_ensembles) and args.generators_dir != args.output_dir:
        logger.warning(
            "--no-prepare-ensembles: set --output-dir to the same directory containing the ensembles (--generators-dir)"
            " unless you already moved the prepared ensembles to --output-dir",
        )
    if args.dataset is None:
        if args.predictor == "Pepsi":
            args.dataset = "/app/peptonebench/datasets/PeptoneDB-SAXS/PeptoneDB-SAXS.csv"
        elif args.predictor in ("UCBshift", "Sparta+", "UCBshift_Sparta+"):
            args.dataset = "/app/peptonebench/datasets/PeptoneDB-CS/PeptoneDB-CS.csv"
        else:
            raise ValueError(f"predictor {args.predictor} not supported")
    db = pd.read_csv(args.dataset, index_col="label")
    if len(args.generators) > 0:
        generators = args.generators.split(",")
    else:
        generators = sorted(
            [d for d in os.listdir(args.generators_dir) if os.path.isdir(os.path.join(args.generators_dir, d))],
        )
    logger.info(f"found generators: {generators}")
    for g in generators:
        processed_dir = str(os.path.join(args.output_dir, g))
        os.makedirs(processed_dir, exist_ok=True)
        process_generator(
            predictor=args.predictor,
            generators_dir=args.generators_dir,
            generator_name=g,
            output_dir=processed_dir,
            db=db,
            nproc=args.nprocs,
            prepare_ensembles=args.prepare_ensembles,
            saxs_data=args.saxs_data,
            pepsi_path=args.pepsi_path,
        )


if __name__ == "__main__":
    main()
