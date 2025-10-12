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

logger_config = {"level": logging.INFO, "format": "%(asctime)s %(levelname)s: %(message)s"}
logging.basicConfig(**logger_config)
logger = logging.getLogger(__name__)

from peptonebench import nmrcs, saxs
from peptonebench.constants import (
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
from peptonebench.reweighting import get_sequence_from_traj

N_JOBS = min(joblib.cpu_count(), int(os.getenv("N_JOBS", "64")))
DEFAULTS = {
    "ess_threshold": 10.0,
    "filter_unphysical_frames": True,
    "plots_dir": "rew_plots",
    "gen_filename": GEN_FILENAME,
    "selected_cs_types": DEFAULT_SELECTED_CS_TYPES,
    "cs_predictor": DEFAULT_CS_PREDICTOR,
    "saxs_predictor": DEFAULT_SAXS_PREDICTOR,
    "bmrb_data": BMRB_DATA,
    "sasbdb_data": SASBDB_DATA,
    "integrative_data": INTEGRATIVE_DATA,
    "db_cs": DB_CS,
    "db_saxs": DB_SAXS,
    "db_integrative": DB_INTEGRATIVE,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reweighting protein ensemble according to experimental data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--generator-dir",
        type=str,
        nargs="+",
        required=True,
        help="folder with generated ensembles to be reweighted",
    )
    parser.add_argument(
        "--gen-filename",
        type=str,
        default=DEFAULTS["gen_filename"],
        help="name pattern of generated ensemble forward models files",
    )
    parser.add_argument(
        "--ess-threshold",
        type=float,
        default=DEFAULTS["ess_threshold"],
        help="ESS threshold. Typically 10% of the number of samples.",
    )
    parser.add_argument(
        "--filter-unphysical-frames",
        action="store_true",
        help="assign NaN weights to unphysical frames from the trajectory, bioemu style",
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

    ## CS specific arguments
    parser.add_argument("--cs-predictor", type=str, default=DEFAULTS["cs_predictor"], help="chemical shift predictor")
    parser.add_argument(
        "--selected-cs-types",
        type=str,
        nargs="+",
        default=DEFAULTS["selected_cs_types"],
        help="list of chemical shifts types to consider. Default is to use all available ('all').",
    )
    parser.add_argument("--bmrb-data", type=str, default=DEFAULTS["bmrb_data"], help="folder with BMRB data")
    parser.add_argument(
        "--db-cs",
        type=str,
        default=DEFAULTS["db_cs"],
        help="path to PeptoneDB-CS.csv file, used for gscores",
    )

    ## SAXS specific arguments
    parser.add_argument("--saxs-predictor", type=str, default=DEFAULTS["saxs_predictor"], help="SAXS predictor")
    parser.add_argument("--sasbdb-data", type=str, default=DEFAULTS["sasbdb_data"], help="path to SASBDB clean data")
    parser.add_argument("--db-saxs", type=str, default=DEFAULTS["db_saxs"], help="path to PeptoneDB-SAXS.csv file")

    ## Integrative specific arguments
    parser.add_argument(
        "--integrative-data",
        type=str,
        default=DEFAULTS["integrative_data"],
        help="path to PeptoneDB-Integrative dataset",
    )
    parser.add_argument(
        "--db-integrative",
        type=str,
        default=DEFAULTS["db_integrative"],
        help="path to PeptoneDB-Integrative.csv file, used for gscores",
    )

    return parser.parse_args()


def cs_reweight_dir(
    generator_dir: str,
    gen_filename: str = DEFAULTS["gen_filename"],
    ess_threshold: float = DEFAULTS["ess_threshold"],
    filter_unphysical_frames: bool = DEFAULTS["filter_unphysical_frames"],
    consistency_check: bool = False,
    plots_dir: str = DEFAULTS["plots_dir"],
    logger_config: dict = None,
    cs_predictor: str = DEFAULTS["cs_predictor"],
    selected_cs_types: list = DEFAULTS["selected_cs_types"],
    bmrb_data: str = DEFAULTS["bmrb_data"],
    db_cs: str = DEFAULTS["db_cs"],
    saxs_predictor: str = DEFAULTS["saxs_predictor"],
    sasbdb_data: str = DEFAULTS["sasbdb_data"],
    db_saxs: str = DEFAULTS["db_saxs"],
    integrative_data: str = DEFAULTS["integrative_data"],
    db_integrative: str = DEFAULTS["db_integrative"],
    n_jobs: int = N_JOBS,
) -> None:
    if selected_cs_types is not None:
        selected_cs_types = sorted(set(selected_cs_types))
    if selected_cs_types == ["all"]:
        selected_cs_types = None
    sel_atom_types_tag = ("-" + "_".join(selected_cs_types)) if selected_cs_types is not None else ""
    filter_tag = "" if filter_unphysical_frames else "-unfiltered"

    for kind, predictor in zip(["cs", "saxs"], [cs_predictor, saxs_predictor]):
        tag = f"-{predictor}{sel_atom_types_tag}{filter_tag}" if kind == "cs" else f"-{predictor}{filter_tag}"
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

        if len(labels[0].split("_")) == 4:  # TODO: check all labels not just the first one
            logger.info("Detected BMRB labels")
            labels = sorted(labels, key=lambda x: (len(x), x.split("_")[0]))
            data_path = bmrb_data
            db_csv = db_cs
        elif labels[0].startswith("SASD"):
            logger.info("Detected SASBDB labels")
            data_path = sasbdb_data
            db_csv = db_saxs
        else:
            logger.info("Detected Integrative labels")
            data_path = integrative_data
            db_csv = db_integrative

        if consistency_check or kind == "cs":
            pep_df = pd.read_csv(os.path.join(data_path, db_csv), index_col="label")
            if consistency_check:
                logger.info(f"Performing consistency check between ensembles and '{db_csv}'")
                for label in labels:
                    assert (
                        get_sequence_from_traj(os.path.join(generator_dir, label)) == pep_df.loc[label, "sequence"]
                    ), f"Consistency check failed for label {label}"
                logger.info("Consistency check passed successfully")
            if kind == "cs":
                gscores_dct = {
                    label: np.asarray(json.loads(pep_df.loc[label, "gscores"]), dtype=float) for label in pep_df.index
                }
                for label in labels:
                    if label not in gscores_dct:
                        logger.error(
                            f"{label} not found in {db_csv}: not using gscores to balance the uncertainties, for all labels"
                        )
                        gscores_dct = dict.fromkeys(labels, value=None)
            del pep_df

        n_jobs = min(n_jobs, len(labels))
        logger.info(f"Processing {len(labels)} labels in parallel using {n_jobs} jobs...")
        if plots_dir:
            os.makedirs(os.path.join(generator_dir, plots_dir + tag), exist_ok=True)

        if kind == "cs":  # FIXME this works but it's ugly

            def reweight_entry(label: str) -> dict:
                return nmrcs.reweight_cs_from_label(
                    label=label,
                    gscores=gscores_dct[label],
                    generator_dir=generator_dir,
                    predictor=cs_predictor,
                    data_path=data_path,
                    selected_cs_types=selected_cs_types,
                    filter_unphysical_frames=filter_unphysical_frames,
                    ess_abs_threshold=ess_threshold,
                    plots_dir=plots_dir + tag,
                    logger_config=logger_config,
                )
        elif kind == "saxs":

            def reweight_entry(label: str) -> dict:
                return saxs.reweight_saxs_from_label(
                    label=label,
                    generator_dir=generator_dir,
                    predictor=saxs_predictor,
                    data_path=data_path,
                    filter_unphysical_frames=filter_unphysical_frames,
                    ess_abs_threshold=ess_threshold,
                    plots_dir=plots_dir + tag,
                    logger_config=logger_config,
                )
        else:
            raise ValueError(f"Unknown kind: {kind}")

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
    # TODO: also save a scatter plot with LOWESS and final score
    logger.info("Done\n")


if __name__ == "__main__":
    args = get_args()
    logger_config["level"] = getattr(logging, args.loglevel)
    logger.setLevel(logger_config["level"])
    del args.loglevel

    for generator_dir in args.generator_dir:
        logger.info(f"---------- Processing directory {generator_dir} ----------")
        args.generator_dir = generator_dir
        try:
            cs_reweight_dir(**vars(args), logger_config=logger_config)
        except Exception:
            logger.error(f"Error processing directory {generator_dir}: {traceback.format_exc()}")
