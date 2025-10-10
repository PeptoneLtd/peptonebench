import joblib
import matplotlib.pyplot as plt
import mdtraj as md
import nglview as nv
import numpy as np
from matplotlib.colors import rgb2hex
from statsmodels.nonparametric.smoothers_lowess import lowess


def get_lowess_fit(
    x: np.ndarray,
    y: np.ndarray,
    xvals: np.ndarray,
    frac: float = 0.5,
    it: int = 1,
    n_bootstrap: int = 200,
    ci: float = 95.0,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit smooth lowess curve to data, with confidence intervals via bootstrapping.
    Parameters
    ----------
    x : np.ndarray
        x data
    y : np.ndarray
        y data
    xvals : np.ndarray
        x values where to evaluate the fit
    frac : float, optional
        The fraction of the data used when estimating each y-value, by default 2/3
    it : int, optional
        The number of residual-based reweightings to perform, by default 3
    n_bootstrap : int, optional
        Number of bootstrap samples to estimate confidence intervals, by default 200
    ci : float, optional
        Confidence interval percentage, by default 95.0
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        yfit, lower_ci, upper_ci"""

    yfit = lowess(y, x, frac, it, xvals=xvals)

    def bootstrap_lowess(i: int) -> np.ndarray:
        idx = np.random.choice(len(x), len(x), replace=True)
        return lowess(y[idx], x[idx], frac, it, xvals=xvals)

    boot_preds = np.array(
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(bootstrap_lowess)(i) for i in range(n_bootstrap)),
    )
    lower_ci = np.percentile(boot_preds, (100 - ci) / 2, axis=0)
    upper_ci = np.percentile(boot_preds, 100 - (100 - ci) / 2, axis=0)
    return yfit, lower_ci, upper_ci


def nv_custom_coloring(
    quantity_per_residue: np.ndarray,
    topology: md.Topology,
    name: str = "custom",
    cmap: str = "coolwarm",
    n_colors: int = 10,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> str:
    """
    nglview per-residue custom coloring according to given quantity, e.g. gscores.
    mdtraj topology is needed to get correct resSeq numbering.
    Example:
        view = nv.show_mdtraj(trj, default_representation=False)
        view.add_cartoon(color=nv_custom_coloring(gscores, trj.top))
        view.center()
    """

    mycmap = plt.get_cmap(cmap, n_colors) if isinstance(cmap, str) else cmap
    colors = [rgb2hex(mycmap(i)) for i in range(n_colors)]
    if vmin is None:
        vmin = np.nanmin(quantity_per_residue)
    if vmax is None:
        vmax = np.nanmax(quantity_per_residue)
    quantity_per_residue = np.clip(quantity_per_residue, vmin, vmax)
    epsilon = (vmax - vmin) / n_colors / 1000
    quantity_range = np.linspace(vmin, vmax + epsilon, n_colors + 1)

    res_per_color = {}
    for i in range(len(quantity_per_residue)):
        if not np.isnan(quantity_per_residue[i]):
            for j in range(n_colors):
                if quantity_per_residue[i] >= quantity_range[j] and quantity_per_residue[i] < quantity_range[j + 1]:
                    res_per_color.setdefault(colors[j], []).append(topology.residue(i).resSeq)
                    break

    nv.color.ColormakerRegistry.add_selection_scheme(
        name,
        [[c, " ".join(str(x) for x in res_per_color[c])] for c in res_per_color],
    )
    return name
