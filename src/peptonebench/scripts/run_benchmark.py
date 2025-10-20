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
import json
import logging
import os
import traceback
from glob import glob

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from peptonebench.config import (
    BMRB_DATA,
    DB_CS,
    DB_INTEGRATIVE,
    DB_SAXS,
    DEFAULT_CS_PREDICTOR,
    DEFAULT_SAXS_PREDICTOR,
    DEFAULT_SELECTED_CS_TYPES,
    GEN_FILENAME,
    INTEGRATIVE_DATA,
    SASBDB_DATA,
)

logger_config = {"level": logging.INFO, "format": "%(asctime)s %(levelname)s: %(message)s"}
logging.basicConfig(**logger_config)
logger = logging.getLogger(__name__)

from peptonebench import nmrcs, reweighting, saxs

N_JOBS = min(joblib.cpu_count(), int(os.getenv("N_JOBS", "64")))
DEFAULTS = {
    "generator_dir": ["."],
    "ess_target": 10.0,
    "filter_unphysical_frames": True,
    "plots_dir": "rew_plots",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Reweight protein ensembles according to experimental data and generate benchmark results.
        The script processes all generated ensembles found in the specified folder(s), expecting the following 
        naming convention:
        [1] LABEL.pdb, LABEL.xtc for the generated ensembles, using the entry label in the PeptoneDB datasets
        [2] PREDICTOR-LABEL.csv for the forward model predictions of the ensembles.
        The script automatically detects the type of entry (PeptoneDB-CS, PeptoneDB-SAXS, or PeptoneDB-Integrative) 
        and reweighting required.
        It is possible to specify the location of the PeptoneDB datasets using the PEPTONEDB_PATH environment variable.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--generator-dir",
        type=str,
        nargs="+",
        default=DEFAULTS["generator_dir"],
        help="folder(s) with generated protein ensembles to be analyzed and reweighted",
    )
    parser.add_argument(
        "--ess-target",
        type=float,
        default=DEFAULTS["ess_target"],
        help="target effective sample size (ESS) for the reweighting. Typically 10%% of the number of samples.",
    )
    parser.add_argument(
        "--filter-unphysical-frames",
        action="store_true",
        help="assign NaN weights to unphysical frames in the ensemble, using BioEmu criteria",
    )
    parser.add_argument("--no-filter-unphysical-frames", dest="filter_unphysical_frames", action="store_false")
    parser.set_defaults(filter_unphysical_frames=DEFAULTS["filter_unphysical_frames"])
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=DEFAULTS["plots_dir"],
        help="directory to save fitting plots. If empty, no plots are saved",
    )
    parser.add_argument(
        "--consistency-check",
        action="store_true",
        help="perform sequence consistency check between generated ensembles and DB entries",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level. NB: it won't affect loops parallelized with joblib",
    )
    parser.add_argument(
        "--gen-filename",
        type=str,
        default=GEN_FILENAME,
        help="name pattern of generated ensemble forward models files",
    )

    ## CS specific arguments
    parser.add_argument(
        "--cs-predictor",
        type=str,
        default=DEFAULT_CS_PREDICTOR,
        help="name of chemical shifts forward-model predictor",
    )
    parser.add_argument(
        "--selected-cs-types",
        type=str,
        nargs="+",
        default=DEFAULT_SELECTED_CS_TYPES,
        help="list of chemical shifts types to consider, set to 'all' or leave None to use all available.",
    )
    parser.add_argument("--bmrb-data", type=str, default=BMRB_DATA, help="folder with BMRB data")
    parser.add_argument(
        "--db-cs",
        type=str,
        default=DB_CS,
        help="path to PeptoneDB-CS.csv file, used for gscores",
    )

    ## SAXS specific arguments
    parser.add_argument(
        "--saxs-predictor",
        type=str,
        default=DEFAULT_SAXS_PREDICTOR,
        help="name of SAXS forward-model predictor",
    )
    parser.add_argument("--sasbdb-data", type=str, default=SASBDB_DATA, help="path to SASBDB clean data")
    parser.add_argument("--db-saxs", type=str, default=DB_SAXS, help="path to PeptoneDB-SAXS.csv file")

    ## Integrative specific arguments
    parser.add_argument(
        "--integrative-data",
        type=str,
        default=INTEGRATIVE_DATA,
        help="path to PeptoneDB-Integrative dataset",
    )
    parser.add_argument(
        "--db-integrative",
        type=str,
        default=DB_INTEGRATIVE,
        help="path to PeptoneDB-Integrative.csv file, used for gscores",
    )
    return parser.parse_args()


def reweight_dir(
    generator_dir: str = DEFAULTS["generator_dir"],
    gen_filename: str = GEN_FILENAME,
    ess_target: float = DEFAULTS["ess_target"],
    filter_unphysical_frames: bool = DEFAULTS["filter_unphysical_frames"],
    consistency_check: bool = False,
    plots_dir: str = DEFAULTS["plots_dir"],
    logger_config: dict = None,
    cs_predictor: str = DEFAULT_CS_PREDICTOR,
    selected_cs_types: list = DEFAULT_SELECTED_CS_TYPES,
    bmrb_data: str = BMRB_DATA,
    db_cs: str = DB_CS,
    saxs_predictor: str = DEFAULT_SAXS_PREDICTOR,
    sasbdb_data: str = SASBDB_DATA,
    db_saxs: str = DB_SAXS,
    integrative_data: str = INTEGRATIVE_DATA,
    db_integrative: str = DB_INTEGRATIVE,
    n_jobs: int = N_JOBS,
) -> None:
    if selected_cs_types is not None:
        selected_cs_types = sorted(set(selected_cs_types))
    if selected_cs_types == ["all"]:
        selected_cs_types = None
    sel_atom_types_tag = ("-" + "_".join(selected_cs_types)) if selected_cs_types is not None else ""
    filter_tag = "" if filter_unphysical_frames else "-unfiltered"

    for kind, predictor in zip(["CS", "SAXS"], [cs_predictor, saxs_predictor]):
        tag = f"-{predictor}{sel_atom_types_tag}{filter_tag}" if kind == "CS" else f"-{predictor}{filter_tag}"
        filenames = os.path.join(
            generator_dir,
            gen_filename.replace("PREDICTOR", predictor).replace("LABEL", "*"),
        )
        prefix, suffix = filenames.split("*")
        labels = sorted([f[len(prefix) : -len(suffix)] for f in glob(filenames)])
        if len(labels) == 0:
            logger.info(
                f"no matching files found in {generator_dir} for: "
                f"{gen_filename.replace('PREDICTOR', predictor).replace('LABEL', '*')}",
            )
            continue

        if len(labels[0].split("_")) == 4:  # checking only the first one, this can be improved
            logger.info("Detected BMRB labels")
            if kind == "SAXS":
                logger.warning("no SAXS data in BMRB, skipping...")
                continue
            labels = sorted(labels, key=lambda x: (len(x), x.split("_")[0]))
            data_path = bmrb_data
            db_csv = db_cs
        elif labels[0].startswith("SASD"):
            logger.info("Detected SASBDB labels")
            if kind == "CS":
                logger.warning("no CS data in SASBDB, skipping...")
                continue
            data_path = sasbdb_data
            db_csv = db_saxs
        else:
            logger.info("Detected Integrative labels")
            data_path = integrative_data
            db_csv = db_integrative

        gscores_dct = None
        if consistency_check or kind == "CS":
            pep_df = pd.read_csv(db_csv, index_col="label")
            if consistency_check:
                logger.info(f"Performing consistency check between ensembles and '{db_csv}'")
                for label in labels:
                    assert (
                        reweighting.get_sequence_from_ensemble(os.path.join(generator_dir, label))
                        == pep_df.loc[label, "sequence"]
                    ), f"Consistency check failed for label {label}"
                logger.info("Consistency check passed successfully")
            if kind == "CS":
                gscores_dct = {
                    label: np.asarray(json.loads(pep_df.loc[label, "gscores"]), dtype=float) for label in pep_df.index
                }
                for label in labels:
                    if label not in gscores_dct:
                        logger.error(
                            f"{label} not found in {db_csv}: "
                            f"NOT using G-scores to balance the uncertainties, for all labels",
                        )
                        gscores_dct = dict.fromkeys(labels, value=None)
            del pep_df

        n_jobs = min(n_jobs, len(labels))
        logger.info(f"Processing {len(labels)} labels in parallel using {n_jobs} jobs...")
        if plots_dir:
            os.makedirs(os.path.join(generator_dir, plots_dir + tag), exist_ok=True)

        def reweight_entry(
            label: str,
            kind: str = kind,
            predictor: str = predictor,
            data_path: str = data_path,
            tag: str = tag,
            gscores_dct: dict = gscores_dct,
        ) -> dict:
            if kind == "CS":
                expt_shift = None
                std_delta = nmrcs.std_delta_cs_from_label(
                    label=label,
                    generator_dir=generator_dir,
                    predictor=predictor,
                    data_path=data_path,
                    selected_cs_types=selected_cs_types,
                    gscores=gscores_dct[label],
                    gen_filename=gen_filename,
                )
            elif kind == "SAXS":
                std_delta, expt_shift = saxs.std_Igen_and_Iexp_from_label(
                    label=label,
                    generator_dir=generator_dir,
                    predictor=predictor,
                    data_path=data_path,
                    gen_filename=gen_filename,
                )
            else:
                raise ValueError(f"Unknown kind: {kind}")
            return reweighting.benchmark_reweighting(
                label=label,
                generator_dir=generator_dir,
                std_delta=std_delta,
                expt_shift=expt_shift,
                filter_unphysical_frames=filter_unphysical_frames,
                ess_target=ess_target,
                plots_dir=plots_dir + tag if plots_dir else "",
                logger_config=logger_config,
            )

        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(reweight_entry)(label) for label in tqdm(labels))

        logger.info("Saving results")
        filename = os.path.join(generator_dir, f"rew_%s{tag}.csv")
        info_columns = [key for key in results[0] if key != "weights"]
        with open(filename % "info", "w") as f:
            f.write(",".join(info_columns) + "\n")
            for row in results:
                f.write(",".join([str(row[col]) for col in info_columns]) + "\n")
        n_samples = len(results[0]["weights"])
        with open(filename % "weights", "w") as f:
            f.write(",".join(labels) + "\n")
            for i in range(n_samples):
                f.write(",".join([str(results[j]["weights"][i]) for j in range(len(labels))]) + "\n")
    logger.info("Done\n")


def main() -> None:
    args = get_args()
    logger_config["level"] = getattr(logging, args.loglevel)
    logger.setLevel(logger_config["level"])
    del args.loglevel

    for generator_dir in args.generator_dir:
        logger.info(f"---------- Processing directory {generator_dir} ----------")
        args.generator_dir = generator_dir
        try:
            reweight_dir(**vars(args), logger_config=logger_config)
        except Exception:
            logger.error(f"Error processing directory {generator_dir}: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
