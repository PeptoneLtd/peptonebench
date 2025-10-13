import logging
import os.path

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_SAXS_PREDICTOR,
    GEN_FILENAME,
    I_SAXS_FILENAME,
    INTEGRATIVE_DATA,
    SASBDB_DATA,
    SASBDB_FILENAME,
)

logger = logging.getLogger(__name__)


def parse_sasbdb_dat(filename: str) -> pd.DataFrame:
    assert filename.endswith(".dat"), "Input file should be a .dat file"
    try:
        expt_df = pd.read_csv(
            filename,
            sep="\\s+",
            header=None,
            names=["q", "I(q)", "sigma"],
            usecols=[0, 1, 2],
            on_bad_lines="skip",
            encoding="latin1",  # SASDJF6 contains non-UTF-8 characters
        )
        expt_df = expt_df.apply(pd.to_numeric, errors="coerce").dropna()
        if len(expt_df) == 0:
            raise pd.errors.ParserError
    except pd.errors.ParserError:  # Try again without sigma
        expt_df = pd.read_csv(
            filename,
            sep="\\s+",
            header=None,
            names=["q", "I(q)"],
            usecols=[0, 1],
            on_bad_lines="skip",
            encoding="latin1",  # SASDJF6 contains non-UTF-8 characters
        )
        expt_df = expt_df.apply(pd.to_numeric, errors="coerce").dropna()
        expt_df["sigma"] = np.full(len(expt_df), np.nan)
    return expt_df


def parse_sasbdb_out(filename: str, rescale_to_dat: bool = True) -> pd.DataFrame:
    assert filename.endswith(".out"), "Input file should be a .out file"
    if not os.path.exists(filename):
        logger.info(f"{filename} does not exist")
        return []
    data = []
    extra_empty_lines = False
    with open(filename, encoding="latin1") as f:
        for line in f:
            if line.strip() == "S          J EXP       ERROR       J REG       I REG" and next(f).strip() == "":
                line_split = next(f).split()
                if len(line_split) == 0:
                    extra_empty_lines = True
                    line_split = next(f).split()
                    line_split = next(f).split()
                while len(line_split) == 2 or len(line_split) == 5:
                    if len(line_split) == 2:
                        data.append((float(line_split[0]), np.nan, np.nan, np.nan, float(line_split[1])))
                    elif len(line_split) == 5:
                        data.append([float(value) for value in line_split])
                    line_split = next(f).split()
                    if extra_empty_lines:
                        line_split = next(f).split()
                break
    data_df = pd.DataFrame(data, columns=["q", "I(q)", "sigma", "Ireg(q)", "Ireg_full(q)"])
    expt_df = data_df[["q", "I(q)", "sigma"]].dropna()
    if len(expt_df) == 0:
        logger.info(f"no data found in {filename}")
        return []
    expt_df.loc[expt_df["sigma"] == -1, "sigma"] = np.nan
    if expt_df.isnull().values.any():
        logger.info(
            f"interpolating missing sigma values ({sum(expt_df.isnull().sum()) / len(expt_df):.2%}) in {filename}",
        )
        expt_df = expt_df.interpolate(method="linear", limit_direction="both")
    if rescale_to_dat:
        # Rescale to match the experimental .dat file
        expt_data_df = parse_sasbdb_dat(filename.replace(".out", ".dat"))
        if len(expt_data_df) == 0:
            normalization_factor = 1.0
        else:
            normalization_factor = (
                expt_df["I(q)"].iloc[0]
                / expt_data_df.loc[(expt_data_df["q"] - expt_df["q"].iloc[0]).abs().idxmin(), "I(q)"]
            )
        expt_df["sigma"] /= normalization_factor
        expt_df["I(q)"] /= normalization_factor
    return expt_df


def get_exp_df_from_label(label: str, data_path: str = None) -> pd.DataFrame:
    if label.startswith("SASD"):
        if data_path is None:
            data_path = SASBDB_DATA
        filename = os.path.join(data_path, SASBDB_FILENAME.replace("LABEL", label))
    else:
        if data_path is None:
            data_path = INTEGRATIVE_DATA
        filename = os.path.join(data_path, I_SAXS_FILENAME.replace("LABEL", label))
    return parse_sasbdb_dat(filename)


def std_Igen_and_Iexp(
    Igen: np.ndarray,  # shape (n_samples, n_obs)
    Iexp: np.ndarray,  # shape (n_obs,)
    sigma: np.ndarray,  # shape (n_obs,)
) -> tuple[np.ndarray, np.ndarray]:
    return Igen / sigma, Iexp / sigma


def std_Igen_and_Iexp_from_label(
    label: str,
    generator_dir: str,
    predictor: str = DEFAULT_SAXS_PREDICTOR,
    data_path: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exp_df = get_exp_df_from_label(label, data_path)
    filename = os.path.join(generator_dir, GEN_FILENAME.replace("LABEL", label).replace("PREDICTOR", predictor))
    gen_df = pd.read_csv(filename, index_col=0)
    # pandas adds .N to duplicate column names, so astype(float) can fail
    assert np.allclose(exp_df["q"], [float(".".join(x.split(".")[:2])) for x in gen_df.columns]), (
        f"Q values do not match: {generator_dir}, {label}"
    )
    return std_Igen_and_Iexp(gen_df.to_numpy(), exp_df["I(q)"].to_numpy(), exp_df["sigma"].to_numpy())
