import logging
import os.path
import subprocess

import mdtraj as md
import numpy as np
import pandas as pd
import pynmrstar
from trizod.potenci.potenci import getpredshifts

from .constants import (
    BMRB_DATA,
    BMRB_FILENAME,
    CS_UNCERTAINTIES,
    DB_CS,
    DEFAULT_CS_PREDICTOR,
    GEN_FILENAME,
    I_CS_FILENAME,
    INTEGRATIVE_DATA,
    ONE_TO_THREE_AA,
    OUTLIERS_FILTER,
    POTENCI_UNCERTAINTIES,
)

logger = logging.getLogger(__name__)


def extended_cs_uncertainties(
    cs_keys: list[tuple[int, str]],
    gscores: list[float] = None,
    ord_unc: dict[str, float] = CS_UNCERTAINTIES[DEFAULT_CS_PREDICTOR],
    disord_unc: dict[str, float] = POTENCI_UNCERTAINTIES,
) -> np.ndarray:
    if gscores is None:
        return np.array([ord_unc[a] for res, a in cs_keys])
    gscores = np.nan_to_num(gscores, nan=0.5)
    return np.array([ord_unc[a] + gscores[res - 1] * (disord_unc[a] - ord_unc[a]) for res, a in cs_keys])


def bmrb_entry_from_label(label: str, data_path: str = BMRB_DATA) -> tuple["pynmrstar.Entry", str, str, str, str]:
    assert len(label.split("_")) == 4, "expected trizod labling convention: entryID_stID_assemID_entityID"
    entryID, stID, assemID, entityID = label.split("_")

    filename = os.path.join(data_path, BMRB_FILENAME.replace("ENTRYID", entryID))
    if not os.path.exists(filename):
        logger.warning(f"{label:>7} - file not found '{filename}', loading from BMRB")
        subprocess.run(
            ["wget", f"https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{entryID}/bmr{entryID}_3.str", "-O", filename],
            check=True,
        )
    entry = pynmrstar.Entry.from_file(filename)
    return entry, entryID, stID, assemID, entityID


def sequence_from_bmrb_entry(entry: pynmrstar.Entry, entityID: str) -> str:
    tags = entry.get_tags(["_Entity.ID", "_Entity.Polymer_seq_one_letter_code"])
    return tags["_Entity.Polymer_seq_one_letter_code"][tags["_Entity.ID"].index(str(entityID))].replace("\n", "")


def sequence_from_bmrb_label(label: str, data_path: str = BMRB_DATA) -> str:
    entry, entryID, stID, assemID, entityID = bmrb_entry_from_label(label, data_path)
    return sequence_from_bmrb_entry(entry, entityID)


def experimental_cs_from_bmrb_label(
    label: str,
    data_path: str = BMRB_DATA,
    filter_outliers: bool = True,
) -> dict[tuple[int, str], float]:
    entry, entryID, stID, assemID, entityID = bmrb_entry_from_label(label, data_path)
    cs = {}
    for chemical_shift_loop in entry.get_loops_by_category("Atom_chem_shift"):
        for line in chemical_shift_loop.get_tag(
            ["Entity_assembly_ID", "Entity_ID", "Assigned_chem_shift_list_ID", "Comp_ID", "Seq_ID", "Atom_ID", "Val"],
        ):
            if line[0] == assemID and line[1] == entityID and line[2] == stID and float(line[6]) > 0:
                if filter_outliers:
                    if line[5] not in POTENCI_UNCERTAINTIES:  # keep only CS types we can handle
                        continue
                    if (line[3], line[5]) in OUTLIERS_FILTER:
                        low, high = OUTLIERS_FILTER[(line[3], line[5])]
                        if not (low < float(line[6]) < high):
                            logger.info(
                                f"{label:>7}: ignoring outlier res_{line[4]} {line[5]}"
                                f" cs={line[6]}, expected range ({low},{high})",
                            )
                            continue
                    else:
                        logger.warning(f"{label:>7}: no outlier filter available for {line[3]} {line[5]}")
                cs[(int(line[4]), line[5])] = float(line[6])
    return cs


def experimental_cs_from_integrative_label(
    label: str,
    data_path: str = INTEGRATIVE_DATA,
    filter_outliers: bool = True,
) -> dict[tuple[int, str], float]:
    filename = os.path.join(data_path, I_CS_FILENAME.replace("LABEL", label))
    cs_types = []
    cs = {}
    with open(filename) as f:
        for line in f:
            line_split = line.split()
            if not cs_types:
                assert len(line_split) >= 2, "experimental CS file header too short"
                assert line_split[0] == "#RESID", "expected '#RESID' as first header column"
                assert line_split[1] == "RESNAME", "expected 'RESNAME' as second header column"
                cs_types = line_split[2:]
                if "HN" in cs_types:
                    cs_types[cs_types.index("HN")] = "H"
            else:
                for i, val in enumerate(line_split[2:]):
                    if val.lower() != "nan" and float(val) > 0:
                        if filter_outliers:
                            if cs_types[i] not in POTENCI_UNCERTAINTIES:  # keep only CS types we can handle
                                continue
                            if len(line_split[1]) == 1:
                                line_split[1] = ONE_TO_THREE_AA.get(line_split[1], line_split[1])
                            if (line_split[1], cs_types[i]) in OUTLIERS_FILTER:
                                low, high = OUTLIERS_FILTER[(line_split[1], cs_types[i])]
                                if not (low < float(val) < high):
                                    logger.info(
                                        f"{label:>7}: ignoring outlier res_{line_split[0]} {cs_types[i]}"
                                        f" cs={val}, expected range ({low},{high})",
                                    )
                                    continue
                            else:
                                logger.warning(
                                    f"{label:>7}: no outlier filter available for {line_split[1]} {cs_types[i]}",
                                )
                        cs[(int(line_split[0]), cs_types[i])] = float(val)
    return cs


def experimental_cs_from_label(
    label: str,
    data_path: str = None,
    filter_outliers: bool = True,
) -> dict[tuple[int, str], float]:
    if label.count("_") == 3:
        data_path = BMRB_DATA if data_path is None else data_path
        return experimental_cs_from_bmrb_label(label, data_path=data_path, filter_outliers=filter_outliers)
    else:
        data_path = INTEGRATIVE_DATA if data_path is None else data_path
        return experimental_cs_from_integrative_label(label, data_path=data_path, filter_outliers=filter_outliers)


def label_trajectory_consistency_check(
    label: str,
    generator_dir: str,
    info: pd.DataFrame,
    trj_ext: str = ".pdb",
) -> bool:
    trj_file = os.path.join(generator_dir, label + trj_ext)
    if not os.path.exists(trj_file):
        old_trj_file = trj_file
        trj_file = os.path.join(generator_dir, label + "-0" + trj_ext)
        if not os.path.exists(trj_file):
            raise FileNotFoundError(f"trajectory file not found: {old_trj_file}, {trj_file}")
    trj_sequence = md.load_frame(trj_file, index=0).top.to_fasta()[0]

    return trj_sequence == info.loc[label, "sequence"]


def read_generated_cs(filename: str) -> pd.DataFrame:  # shape (n_samples, n_obs)
    """Read generated chemical shifts from file."""
    assert filename.endswith(".csv"), "expected .csv file"
    with open(filename) as f:
        resSeq, name0 = f.readline().split(",")[:2]
        name1 = f.readline().split(",")[0]
    if resSeq != "resSeq" and resSeq != "ResSeq":
        raise ValueError(f"Unsupported Chemical Shifts file: {filename}")
    if name0 == "name":
        df = pd.read_csv(filename, index_col=[0, 1]).T
    elif name1 == "name":
        # commenting "f" because some files have a useless row that stats with "frame"
        df = pd.read_csv(filename, header=[0, 1], index_col=0, comment="f")
        df.columns = df.columns.set_levels([df.columns.levels[0].astype(int), df.columns.levels[1].astype(str)])
        logger.warning(f"fixing format of {filename}")
        df.T.to_csv(filename, index=False)  # overwrite the file with the correct format
    else:
        raise ValueError(f"Unsupported Chemical Shifts file: {filename}")
    return df


def generated_cs_from_label(
    label: str,
    generator_dir: str,
    predictor: str = DEFAULT_CS_PREDICTOR,
) -> pd.DataFrame:  # shape (n_samples, n_obs)
    """Get chemical shifts for a specific generator and label."""
    filename = os.path.join(generator_dir, GEN_FILENAME.replace("LABEL", label).replace("PREDICTOR", predictor))
    return read_generated_cs(filename)


def std_delta_cs(
    gen_cs: pd.DataFrame,
    exp_cs: dict[tuple[int, str], float],
    selected_cs_types: list[str] = None,
    cs_uncertainties: dict[str, float] = CS_UNCERTAINTIES[DEFAULT_CS_PREDICTOR],
    gscores: np.ndarray = None,
    return_keys: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, str]]]:  # shape (n_samples, n_obs)
    """Return generated chemical shifts, standardized with corresponding experimental measurements."""

    shared_keys = sorted(set(exp_cs.keys()).intersection(set(gen_cs.keys())))
    if selected_cs_types is None:
        selected_cs_types = list(cs_uncertainties.keys())
    shared_keys = [(res, a) for res, a in shared_keys if a in selected_cs_types]
    if len(shared_keys) == 0:
        std_delta_cs = np.full((len(gen_cs), 0), np.nan)
    else:
        uncertainties_arr = extended_cs_uncertainties(shared_keys, gscores, ord_unc=cs_uncertainties)
        std_delta_cs = (
            gen_cs[list(shared_keys)].to_numpy() - np.array([exp_cs[k] for k in shared_keys])
        ) / uncertainties_arr

    if return_keys:
        return std_delta_cs, shared_keys
    return std_delta_cs


def std_delta_cs_from_label(
    label: str,
    generator_dir: str,
    predictor: str = DEFAULT_CS_PREDICTOR,
    data_path: str = None,
    selected_cs_types: list[str] = None,
    gscores: np.ndarray = None,
    return_keys: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, str]]]:  # shape (n_samples, n_obs)
    """Return generated chemical shifts, standardized with corresponding experimental measurements."""
    return std_delta_cs(
        gen_cs=generated_cs_from_label(label, generator_dir, predictor),
        exp_cs=experimental_cs_from_label(label, data_path),
        selected_cs_types=selected_cs_types,
        cs_uncertainties=CS_UNCERTAINTIES[predictor],
        gscores=gscores,
        return_keys=return_keys,
    )


def compute_random_coil_cs(
    sequence: str,
    temperature: float = 298.0,
    pH: float = 7.0,
    ionic_strength: float = 0.1,
) -> dict[tuple[int, str], float]:
    """Compute random coil chemical shifts with POTENCI."""
    dct_cs = getpredshifts(sequence, temperature, pH, ionic_strength, pkacsvfile=False)
    return {(k[0], kk): vv for k, v in dct_cs.items() for kk, vv in v.items()}


def random_coil_cs_from_bmrb_label(
    label: str,
    db_cs: str = DB_CS,
    data_path: str = BMRB_DATA,
) -> dict[tuple[int, str], float]:
    """Use info from TRIZOD (or BMRB entry as fallback) to compute random coil chemical shifts with POTENCI."""
    conditions = {}
    if os.path.exists(db_cs):
        df = pd.read_csv(db_cs, index_col="label")
        sequence = df.loc[label, "sequence"]
        conditions["temperature"] = df.loc[label, "temperature"]
        conditions["pH"] = df.loc[label, "pH"]
        conditions["ionic_strength"] = df.loc[label, "ionic_strength(M)"]
    else:
        logger.info(f"PeptoneDB-CS file not found '{db_cs}', using BMRB entry for conditions")
        entry, entryID, stID, assemID, entityID = bmrb_entry_from_label(label, data_path)
        sequence = sequence_from_bmrb_entry(entry, entityID)
        ionic_conversion = {"M": 1, "mM": 1e-3}
        Sample_condition_list_ID = "1"  # FIXME get the correct one!
        for loop in entry.get_loops_by_category("Sample_condition_variable"):
            for line in loop.get_tag(["Sample_condition_list_ID", "Type", "Val", "Val_units"]):
                if line[0] != Sample_condition_list_ID:
                    raise NotImplementedError("multiple Sample_condition_list_ID not supported")
                    continue
                if line[1] in ["temperature", "pH"]:
                    conditions[line[1]] = float(line[2])
                elif line[1] == "ionic strength":
                    val = float(line[2]) * ionic_conversion.get(line[3], 1e-3)
                    while val > 5:
                        logger.warning(f"{label:>7} - suspiciously high ionic strength {val} M, assuming wrong units")
                        val *= 1e-3
                    conditions["ionic_strength"] = val
    return compute_random_coil_cs(sequence, **conditions)


def secondary_cs(cs: dict, conditions: dict, cs_type: str = "CA") -> tuple[np.ndarray, np.ndarray]:
    rc_cs = compute_random_coil_cs(
        sequence=conditions["sequence"],
        temperature=conditions["temperature"],
        pH=conditions["pH"],
        ionic_strength=conditions["ionic_strength(M)"],
    )
    rc_cs = {k: v for k, v in rc_cs.items() if k[1] == cs_type}
    shared_keys = sorted(set(cs.keys()).intersection(set(rc_cs.keys())))
    return np.array([res for res, a in shared_keys]), np.array(
        [cs[(res, a)] - rc_cs[(res, a)] for res, a in shared_keys],
    )
