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

from peptonebench import nmrcs
from peptonebench.constants import (
    BMRB_DATA,
    DB_CS,
    DB_INTEGRATIVE,
    DEFAULT_PREDICTOR,
    DEFAULT_SELECTED_CS_TYPES,
    INTEGRATIVE_DATA,
)
from peptonebench.reweighting import get_sequence_from_traj

N_JOBS = min(joblib.cpu_count(), int(os.getenv("N_JOBS", "64")))
DEFAULTS = {
    "selected_cs_types": DEFAULT_SELECTED_CS_TYPES,
    "predictor": DEFAULT_PREDICTOR,
    "ess_abs_threshold": 10.0,
    "ess_rel_threshold": 0.1,
    "filter_unphysical_frames": True,
    "plots_dir": "rew_plots",
    "bmrb_data": BMRB_DATA,
    "db_cs": DB_CS,
    "integrative_data": INTEGRATIVE_DATA,
    "db_integrative": DB_INTEGRATIVE,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reweighting protein ensemble according to experimental chemical shifts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--generator_dir",
        type=str,
        nargs="+",
        required=True,
        help="folder with generated ensembles to be reweighted",
    )
    parser.add_argument("--predictor", type=str, default=DEFAULTS["predictor"], help="chemical shift predictor")
    parser.add_argument(
        "--selected_cs_types",
        type=str,
        nargs="+",
        default=DEFAULTS["selected_cs_types"],
        help="list of chemical shifts types to consider. Default is to use all available ('all').",
    )
    parser.add_argument(
        "--ess_abs_threshold",
        type=float,
        default=DEFAULTS["ess_abs_threshold"],
        help="absolute ESS threshold",
    )
    parser.add_argument(
        "--ess_rel_threshold",
        type=float,
        default=DEFAULTS["ess_rel_threshold"],
        help="relative ESS threshold",
    )
    parser.add_argument(
        "--filter_unphysical_frames",
        action="store_true",
        help="assign NaN weights to unphysical frames from the trajectory, bioemu style",
    )
    parser.add_argument("--no-filter_unphysical_frames", dest="filter_unphysical_frames", action="store_false")
    parser.set_defaults(filter_unphysical_frames=DEFAULTS["filter_unphysical_frames"])
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=DEFAULTS["plots_dir"],
        help="directory to save fitting plots. If empty, no plots are saved",
    )
    parser.add_argument("--consistency_check", action="store_true", help="perform trajectory consistency check")
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level. NB: it won't affect loops parallelized with joblib",
    )

    ## BMRB specific arguments
    parser.add_argument("--bmrb_data", type=str, default=DEFAULTS["bmrb_data"], help="folder with BMRB data")
    parser.add_argument("--db_cs", type=str, default=DEFAULTS["db_cs"], help="path to gscores file")

    ## Integrative specific arguments
    parser.add_argument(
        "--integrative_data", type=str, default=DEFAULTS["integrative_data"], help="path to high-quality dataset"
    )
    parser.add_argument("--db_integrative", type=str, default=DEFAULTS["db_integrative"], help="path to gscores file")

    return parser.parse_args()


def cs_reweight_dir(
    generator_dir: str,
    bmrb_data: str = DEFAULTS["bmrb_data"],
    db_cs: str = DEFAULTS["db_cs"],
    integrative_data: str = DEFAULTS["integrative_data"],
    db_integrative: str = DEFAULTS["db_integrative"],
    selected_cs_types: list = DEFAULTS["selected_cs_types"],
    predictor: str = DEFAULTS["predictor"],
    ess_abs_threshold: float = DEFAULTS["ess_abs_threshold"],
    ess_rel_threshold: float = DEFAULTS["ess_rel_threshold"],
    filter_unphysical_frames: bool = DEFAULTS["filter_unphysical_frames"],
    consistency_check: bool = False,
    plots_dir: str = DEFAULTS["plots_dir"],
    logger_config: dict = None,
    n_jobs: int = N_JOBS,
) -> None:
    if selected_cs_types is not None:
        selected_cs_types = sorted(set(selected_cs_types))
    if selected_cs_types == ["all"]:
        selected_cs_types = None
    sel_atom_types_tag = ("-" + "_".join(selected_cs_types)) if selected_cs_types is not None else ""
    filter_tag = "-filtered" if filter_unphysical_frames else ""
    tag = f"-{predictor}{sel_atom_types_tag}{filter_tag}-new"

    filenames = os.path.join(
        generator_dir,
        nmrcs.CS_FILENAME.replace("PREDICTOR", predictor).replace("LABEL", "*"),
    )
    prefix, suffix = filenames.split("*")
    labels = sorted([f[len(prefix) : -len(suffix)] for f in glob(filenames)])
    if len(labels) == 0:
        raise ValueError(f"no labels found in {generator_dir}")

    if len(labels[0].split("_")) == 4:
        logger.info("Detected BMRB labels")
        labels = sorted(labels, key=lambda x: (len(x), x.split("_")[0]))
        data_path = bmrb_data
        gscores_csv = db_cs
    else:
        logger.info("Detected Integrative labels")
        if ess_abs_threshold != 100: #FIXME not very robust
            logger.warning("For Integrative data, ess_abs_threshold is forced to be 100")
            ess_abs_threshold = 100
        data_path = integrative_data
        gscores_csv = db_integrative
    info_df = pd.read_csv(os.path.join(data_path, gscores_csv), index_col="label")
    if consistency_check:
        logger.info(f"Performing consistency check between trajectories and '{gscores_csv}'")
        for label in labels:
            assert nmrcs.get_sequence_from_traj(os.path.join(generator_dir, label)) == info_df.loc[label, "sequence"], (
                f"Consistency check failed for label {label}"
            )
        logger.info("Consistency check passed successfully")
    gscores_dct = {
        label: np.asanyarray(json.loads(info_df.loc[label, "gscores"]), dtype=float) for label in info_df.index
    }
    del info_df

    n_jobs = min(n_jobs, len(labels))
    logger.info(f"Processing {len(labels)} labels in parallel using {n_jobs} jobs...")
    if plots_dir:
        plots_dir += tag
        os.makedirs(os.path.join(generator_dir, plots_dir), exist_ok=True)

    def reweight_entry(label: str) -> dict:
        return nmrcs.reweight_cs_from_label(
            label=label,
            gscores=gscores_dct[label],
            generator_dir=generator_dir,
            predictor=predictor,
            data_path=data_path,
            selected_cs_types=selected_cs_types,
            filter_unphysical_frames=filter_unphysical_frames,
            ess_abs_threshold=ess_abs_threshold,
            ess_rel_threshold=ess_rel_threshold,
            plots_dir=plots_dir,
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
