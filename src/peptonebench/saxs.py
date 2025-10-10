import logging
import os.path
import traceback

import numpy as np
import pandas as pd

from . import reweighting
from .constants import I_SAXS_FILENAME, INTEGRATIVE_DATA, SASBDB_DATA, SASBDB_FILENAME, SAXS_FILENAME

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
            f"interpolating missing sigma values ({sum(expt_df.isnull().sum()) / len(expt_df):.2%}) in {filename}"
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
    data_path: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exp_df = get_exp_df_from_label(label, data_path)
    gen_df = pd.read_csv(f"{generator_dir}/{SAXS_FILENAME.replace('LABEL', label)}", index_col=0)
    # pandas adds .N to duplicate column names, so astype(float) can fail
    assert np.allclose(exp_df["q"], [float(".".join(x.split(".")[:2])) for x in gen_df.columns]), (
        f"Q values do not match: {generator_dir}, {label}"
    )
    return std_Igen_and_Iexp(gen_df.to_numpy(), exp_df["I(q)"].to_numpy(), exp_df["sigma"].to_numpy())


def reweight_saxs(
    label: str,
    generator_dir: str,
    std_Igen: np.ndarray,  # shape (n_samples, n_obs)
    std_Iexp: np.ndarray,  # shape (n_obs,)
    filter_unphysical_frames: bool = False,
    ess_abs_threshold: float = 10.0,
    ess_rel_threshold: float = 0.0,
    plots_dir: str = "",
    logger_config: dict = None,
) -> dict:
    """Reweight ensemble for given saxs data"""
    if logger_config:
        logging.basicConfig(**logger_config)
    n_samples, n_obs = std_Igen.shape
    results = {
        "label": label,
        "n_obs": n_obs,
        "n_samples": n_samples,
        "RMSE": np.nan,
        "ESS": np.nan,
        "rew_RMSE": np.nan,
        "min_ESS": np.nan,
        "min_RMSE": np.nan,
        "weights": np.full(n_samples, np.nan),
    }
    if len(std_Igen) == 0:
        logger.warning(f"{label:>7} - skipping, no samples")
        return results
    assert np.isfinite(std_Igen).all(), f"{label:>7} - unexpected NaN values in std_Igen"
    assert np.isfinite(std_Iexp).all(), f"{label:>7} - unexpected NaN values in std_Iexp"
    if filter_unphysical_frames:
        nan_mask = reweighting.get_physical_frames_mask(os.path.join(generator_dir, label))
        logger.info(f"{label:>7} - traj filtered from {n_samples} to {sum(nan_mask)} samples")
        assert len(nan_mask) == n_samples, f"{label:>7} - Mismatch length between trajectory and chemical shifts"
    else:
        nan_mask = np.full(n_samples, True, dtype=bool)
    results["n_samples"] = sum(nan_mask)
    results["RMSE"] = reweighting.get_RMSE(std_Igen[nan_mask], std_Iexp)
    if sum(nan_mask) == 0:
        logger.warning(f"{label:>7} - no valid samples after filtering")
        return results
    target_ess = max(ess_rel_threshold * n_samples, ess_abs_threshold)  # rel threshold is w.r.t. original n_samples
    try:
        res = reweighting.reweight_to_ess(
            # reweighting.run_gamma_minimization,
            reweighting.run_loss_minimization,  # faster but less reliable
            std_delta=std_Igen[nan_mask],
            expt_shift=std_Iexp,
            ess_abs_threshold=target_ess,
            label=label,
        )
    except Exception:
        logger.error(f"{label:>7} - reweighting failed, {traceback.format_exc()}")
        return results
    if plots_dir:
        reweighting.plot_reweighting_results(
            res,
            title=f"SAXS - {os.path.basename(generator_dir)}, {label}\nn_obs={n_obs:_}, n_samples={n_samples:_}, valid_samples={sum(nan_mask):_}",
            filename=os.path.join(generator_dir, plots_dir, f"{label}.png"),
            ess_threshold=ess_abs_threshold,
        )
    results["ESS"] = res["ess"][-1]
    results["rew_RMSE"] = res["rmse"][-1]
    results["min_ESS"] = np.nanmin(res["ess"])
    results["min_RMSE"] = np.nanmin(res["rmse"])
    results["weights"][nan_mask] = res["weights"]

    return results


def reweight_saxs_from_label(
    label: str,
    generator_dir: str,
    data_path: str = None,
    filter_unphysical_frames: bool = False,
    ess_abs_threshold: float = 10.0,
    ess_rel_threshold: float = 0.0,
    plots_dir: str = "",
    logger_config: dict = None,
) -> dict:
    """Reweight a single entry in the saxs dataset."""
    std_Igen, std_Iexp = std_Igen_and_Iexp_from_label(
        label=label,
        generator_dir=generator_dir,
        data_path=data_path,
    )

    return reweight_saxs(
        label=label,
        generator_dir=generator_dir,
        std_Igen=std_Igen,
        std_Iexp=std_Iexp,
        filter_unphysical_frames=filter_unphysical_frames,
        ess_abs_threshold=ess_abs_threshold,
        ess_rel_threshold=ess_rel_threshold,
        plots_dir=plots_dir,
        logger_config=logger_config,
    )
