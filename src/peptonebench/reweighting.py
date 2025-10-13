import logging
import os.path
import traceback
from collections.abc import Callable
from functools import lru_cache

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, root_scalar
from scipy.special import logsumexp, softmax

from .filter_unphysical_samples import _get_physical_traj_indices

logger = logging.getLogger(__name__)


def get_RMSE(
    std_delta: np.ndarray,  # shape (n_samples, n_obs)
    expt_shift: np.ndarray = None,  # shape (n_obs,)
    weights: np.ndarray = None,  # shape (n_samples,)
) -> float:
    """Root Mean Square Error."""
    if std_delta.shape[1] == 0:
        return np.nan
    if weights is None:
        mean = np.average(std_delta, axis=0)
    else:
        mask_nan = np.isfinite(weights)
        if not mask_nan.any():
            return np.nan
        mean = np.average(std_delta[mask_nan], weights=weights[mask_nan], axis=0)
    if expt_shift is not None:
        c = np.sum(expt_shift * mean) / np.sum(mean**2)  # Svergun et al. (1995)
        mean = c * mean - expt_shift
    return np.linalg.norm(mean) / np.sqrt(len(mean))


def get_ESS(weights: np.ndarray) -> float:
    """Kish effective sample size."""
    mask_nan = np.isfinite(weights)
    if not mask_nan.any():
        return np.nan
    return np.square(np.sum(weights[mask_nan])) / np.sum(np.square(weights[mask_nan]))


def run_gamma_minimization(
    theta: float,
    std_delta: np.ndarray,  # shape (n_samples, n_obs)
    expt_shift: np.ndarray = None,  # shape (n_obs,)
    random_x0: bool = True,
    x0: np.ndarray = None,
) -> np.ndarray:
    """Minimize Gamma function to get weights. Convenient when n_samples >> n_obs."""

    if expt_shift is not None:
        raise ValueError("Experimental shift is not supported in gamma minimization.")

    def gamma_and_jac(
        lmbd: np.ndarray,  # shape (n_obs,)
        theta: float = theta,
        std_delta: np.ndarray = std_delta,
    ) -> tuple[float, np.ndarray]:
        logw = -(std_delta @ lmbd)
        gamma = logsumexp(logw) + 0.5 * theta * (lmbd @ lmbd)
        jac = -(softmax(logw) @ std_delta) + theta * lmbd
        rebalance = min(1, 1 / theta)
        return gamma * rebalance, jac * rebalance

    n_obs = std_delta.shape[1]
    n_samples = std_delta.shape[0]
    if x0 is None:
        x0 = np.random.rand(n_obs) if random_x0 else np.zeros(n_obs)
    # res = minimize(fun=gamma_and_jac, x0=x0, jac=True, method="L-BFGS-B") #faster, less accurate
    res = minimize(fun=gamma_and_jac, x0=x0, jac=True, method="BFGS", options={"gtol": 1e-4, "maxiter": 10_000})
    if not res.success:
        logger.warning(f"  theta={theta:e}, failed: {res.message}")
        return np.full(n_samples, np.nan), np.full_like(res.x, np.nan)

    weights = softmax(-std_delta @ res.x)
    return weights, res.x


def run_loss_minimization(
    theta: float,
    std_delta: np.ndarray,  # shape (n_samples, n_obs)
    expt_shift: np.ndarray = None,  # shape (n_obs,)
    random_x0: bool = True,
    x0: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize loss function to get weights. Convenient when n_obs > n_samples.
    Supports the case of arbitrary intensity units, via expt_shift."""

    def loss_and_jac(
        logw: np.ndarray,  # shape (n_obs,)
        theta: float = theta,
        std_delta: np.ndarray = std_delta,
        expt_shift: np.ndarray = expt_shift,
    ) -> tuple[float, np.ndarray]:
        log_weights = logw - logsumexp(logw)
        weights = np.exp(log_weights)
        av_std_delta = weights @ std_delta
        if expt_shift is not None:
            c = np.sum(expt_shift * av_std_delta) / np.sum(av_std_delta**2)  # Svergun et al. (1995)
            av_std_delta = c * av_std_delta - expt_shift
        else:
            c = 1.0

        loss = 0.5 * np.sum(np.square(av_std_delta)) + theta * np.sum(weights * log_weights)

        d_loss_d_weights = std_delta @ (c * av_std_delta) + theta * (1 + log_weights)
        jac = weights * d_loss_d_weights - weights * (weights @ d_loss_d_weights)

        return loss, jac  # NB: no need to adjust jac for the logsumexp shift

    n_samples = std_delta.shape[0]
    if x0 is None:
        x0 = np.random.rand(n_samples) if random_x0 else np.ones(n_samples)
        x0 = np.log(x0 / np.sum(x0))
    # res = minimize(fun=loss_and_jac, x0=x0, jac=True, method="L-BFGS-B") # converges to wrong stuff
    # res = minimize(fun=loss_and_jac, x0=x0, jac=True, method="BFGS", options={"gtol": 1e-4}) # fails almost always
    trust_constr_options = {  # works best, still not great
        "maxiter": 50_000,
        "gtol": 1e-4,
        # "xtol": 1e-5,
        "initial_tr_radius": 0.1,
    }
    res = minimize(fun=loss_and_jac, x0=x0, jac=True, method="trust-constr", options=trust_constr_options)
    ## try with pytorch? https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
    if not res.success:
        logger.warning(f"  theta={theta:e}, failed: {res.message}")
        return np.full(n_samples, np.nan), np.full_like(res.x, np.nan)

    weights = softmax(res.x)
    return weights, res.x


def reweight_to_ess(
    minimization: Callable,
    std_delta: np.ndarray,  # shape (n_samples, n_obs)
    expt_shift: np.ndarray = None,  # shape (n_obs,)
    ess_abs_threshold: float = 10.0,
    ess_rel_threshold: float = 0.0,
    ess_tol: float = 0.1,
    log10_theta_bracket: tuple[float, float] = (-3.0, 8.0),
    n_thetas: int = 5,
    label: str = "  ",  # only for logging
) -> dict:
    """Reweight to a target effective sample size (ESS)."""

    n_samples = std_delta.shape[0]
    assert 0 <= ess_rel_threshold <= 1, f"invalid ess_rel_threshold: {ess_rel_threshold}"
    assert ess_abs_threshold >= 1, f"invalid ess_abs_threshold: {ess_abs_threshold}"
    target_ess = max(ess_rel_threshold * n_samples, ess_abs_threshold)
    if target_ess >= n_samples:
        logger.info(f"{label:>7} - no reweighting performed since target_ess={target_ess} >= n_samples={n_samples}")
        return {
            "log10_theta": list(log10_theta_bracket),
            "ess": 2 * [n_samples],
            "rmse": 2 * [get_RMSE(std_delta, expt_shift)],
            "weights": np.ones(n_samples) / n_samples,
            "x0": None,
        }
    logger.debug(f"{label:>7} - target_ess={target_ess}")
    results = {"log10_theta": [], "ess": [], "rmse": [], "weights": None, "x0": None}

    @lru_cache
    def objective(log10_theta: float, random_x0: bool = False) -> float:
        weights, x0 = minimization(10**log10_theta, std_delta, expt_shift, random_x0, x0=results["x0"])
        if np.isnan(weights).all():
            if results["x0"] is not None:
                logger.warning(f"{label:>7} - minimization failed, retrying with random x0")
                results["x0"] = None
                return objective(log10_theta, random_x0=True)
            else:
                logger.warning(f"{label:>7} - minimization failed")
                return np.nan
        results["x0"] = x0
        results["weights"] = weights
        ess = get_ESS(weights)
        rmse = get_RMSE(std_delta, expt_shift, weights)
        results["log10_theta"].append(log10_theta)
        results["ess"].append(ess)
        results["rmse"].append(rmse)
        if ess > target_ess and ess < target_ess + ess_tol:
            logger.debug(f"{label:>7} - early stopping, ess_tol={ess_tol} reached")
            return 0.0
        return ess - target_ess

    for log10_theta in np.linspace(log10_theta_bracket[1], log10_theta_bracket[0], n_thetas):
        objective(log10_theta)
    if np.nanmin(results["ess"]) > target_ess:
        return results

    results["x0"] = None
    assert np.nanmax(results["ess"]) > target_ess, "please increase log10_theta_bracket upper bound"
    assert np.nanmin(results["ess"]) < target_ess, "please decrease log10_theta_bracket lower bound"
    low = np.arange(len(results["ess"]))[np.array(results["ess"]) < target_ess][0]
    high = np.arange(len(results["ess"]))[np.array(results["ess"]) > target_ess][-1]
    new_brackets = results["log10_theta"][low], results["log10_theta"][high]
    logger.debug(f"{label:>7} - root finding in ({10 ** new_brackets[0]:.3e}, {10 ** new_brackets[1]:.3e})")
    res = root_scalar(f=objective, bracket=new_brackets)
    if not res.converged:
        logger.warning(f"{label:>7} - root finding did not converge, {res.flag}")
    return results


def plot_reweighting_results(results: dict, title: str = "", ess_target: float = 10.0, filename: str = None) -> None:
    """Plot the results of the reweighting, RMSE and ESS as a function of theta."""

    log10_theta_range = np.linspace(min(results["log10_theta"]), max(results["log10_theta"]), 100)
    sorting_order = np.argsort(results["log10_theta"])
    sorting_order = sorting_order[np.isfinite(np.array(results["ess"])[sorting_order])]
    log10_theta = np.array(results["log10_theta"])[sorting_order]
    ess = np.array(results["ess"])[sorting_order]
    rmse = np.array(results["rmse"])[sorting_order]

    plt.figure(figsize=(5, 3))
    plt.title(title)
    plt.plot(
        10**log10_theta_range,
        PchipInterpolator(log10_theta, rmse, extrapolate=False)(log10_theta_range),
        "-",
        color="C0",
    )
    plt.plot(10**log10_theta, rmse, "+", color="C0")
    plt.xscale("log")
    plt.xlim(10 ** (log10_theta_range[0] - 0.5), 10 ** (log10_theta_range[-1] + 0.5))
    plt.xlabel("theta")
    plt.ylabel("RMSE [-]")
    plt.twinx()
    plt.plot([np.nan], "-", label=f"min RMSE = {min(rmse):.2f}")
    plt.plot(
        10**log10_theta_range,
        PchipInterpolator(log10_theta, ess, extrapolate=False)(log10_theta_range),
        "--",
        color="C1",
        label=f"min ESS = {min(ess):.2f}",
    )
    plt.plot(10**log10_theta, ess, "x", color="C1")
    plt.axvline(
        10 ** results["log10_theta"][-1],
        color="k",
        ls=":",
        label=f"RMSE={results['rmse'][-1]:.2f}\nESS={results['ess'][-1]:.2f}",
    )
    if ess_target is not None:
        plt.axhspan(0, ess_target, color="k", alpha=0.1)
    plt.legend()
    plt.ylabel("ESS [--]")
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_ens_top_filenames(ens_filename: str) -> tuple[str, str]:
    """Assuming ensembles in xtc+pdb, pdb, or h5 format.
    ens_filename can be given without extension."""
    top_filename = None
    if len(os.path.basename(ens_filename).split(".")) == 1:
        if os.path.exists(ens_filename + ".xtc"):
            ens_filename += ".xtc"
        else:
            ens_filename += ".pdb"
    if ens_filename.endswith(".xtc"):
        top_filename = ens_filename.replace(".xtc", ".pdb")
    return ens_filename, top_filename


def get_physical_samples_mask(ens_filename: str) -> np.ndarray:
    """Assuming ensembles in xtc+pdb, pdb, or h5 format.
    ens_filename can be given without extension."""
    ens_filename, top_filename = get_ens_top_filenames(ens_filename)
    ens = md.load(ens_filename, top=top_filename)

    mask = np.zeros(len(ens), dtype=bool)
    mask[_get_physical_traj_indices(ens)] = True
    return mask


def get_sequence_from_ensemble(ens_filename: str) -> str:
    """Assuming ensembles in xtc+pdb, pdb, or h5 format.
    ens_filename can be given without extension."""

    ens_filename, top_filename = get_ens_top_filenames(ens_filename)
    ens = md.load_frame(ens_filename, top=top_filename, index=0)

    return ens.top.to_fasta()[0]


def benchmark_reweighting(
    label: str,
    generator_dir: str,
    std_delta: np.ndarray,  # shape (n_samples, n_obs)
    expt_shift: np.ndarray = None,  # shape (n_obs,)
    filter_unphysical_frames: bool = True,
    ess_target: float = 10.0,
    plots_dir: str = "",
    minimization: Callable = None,
    logger_config: dict = None,
) -> dict:
    """Reweight ensembles to generate PeptoneBench results"""
    if logger_config:
        logging.basicConfig(**logger_config)
    n_samples, n_obs = std_delta.shape
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
    if n_samples == 0:
        logger.warning(f"{label:>7} - skipping, no samples")
        return results
    nan_mask = np.isfinite(std_delta).all(axis=1)
    tot_invalid_samples = sum(~nan_mask)
    if tot_invalid_samples > 0:
        logger.info(f"{label:>7} - {tot_invalid_samples} samples were discarded due to NaN in forward model data")
    if filter_unphysical_frames:
        valid_frames = get_physical_samples_mask(os.path.join(generator_dir, label))
        logger.info(f"{label:>7} - ensemble filtered from {n_samples} to {sum(valid_frames)} samples")
        assert len(valid_frames) == n_samples, f"{label:>7} - Mismatch length between ensemble and forward model data"
        nan_mask = nan_mask & valid_frames
    results["n_samples"] = sum(nan_mask)
    results["RMSE"] = get_RMSE(std_delta[nan_mask])
    if sum(nan_mask) == 0:
        logger.warning(f"{label:>7} - no valid samples after filtering")
        return results
    if minimization is None:
        minimization = run_gamma_minimization if expt_shift is None else run_loss_minimization
        logger.debug(f"{label:>7} - using {'gamma' if expt_shift is None else 'loss'} minimization")
    try:
        res = reweight_to_ess(
            minimization,
            std_delta=std_delta[nan_mask],
            expt_shift=expt_shift,
            ess_abs_threshold=ess_target,
            label=label,
        )
    except Exception:
        logger.error(f"{label:>7} - reweighting failed, {traceback.format_exc()}")
        return results
    if plots_dir:
        kind = "CS" if expt_shift is None else "SAXS"  # currently only two supported
        generator = f"{os.path.basename(generator_dir)}, " if generator_dir != "." else ""
        plot_reweighting_results(
            res,
            title=f"{kind} - {generator}{label}\n"
            f"n_obs={n_obs:_}, n_samples={n_samples:_}, valid_samples={sum(nan_mask):_}",
            filename=os.path.join(generator_dir, plots_dir, f"{label}.png"),
            ess_target=ess_target,
        )
    results["ESS"] = res["ess"][-1]
    results["rew_RMSE"] = res["rmse"][-1]
    results["min_ESS"] = np.nanmin(res["ess"])
    results["min_RMSE"] = np.nanmin(res["rmse"])
    results["weights"][nan_mask] = res["weights"]

    return results
