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
import traceback
from typing import Tuple

import CSpred as UCBshift
import mdtraj as md
import pandas as pd
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper for Chemical Shift forward model predictors that can handle ensembles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectory", type=str, required=True, help="Trajectory file (e.g. .xtc)")
    parser.add_argument("--topology", type=str, required=True, help="Topology file (e.g. .pdb)")
    parser.add_argument(
        "--predictor",
        type=str,
        nargs="+",
        default=["UCBshift"],
        help="Chemical Shift forward model to use, 'UCBshift' and/or 'Sparta+'",
    )
    parser.add_argument(
        "--pH",
        type=float,
        default=7.0,
        help="use PDBFixer to add hydrogens with proper naming at given pH (set to 0 to skip)",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory for results")
    parser.add_argument("--sequence", default="", help="Provide the aminoacid sequence to check for consistency")
    parser.add_argument("--tmpdir", type=str, default="/tmp", help="Name of temporary directory for results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    return parser.parse_args()


def run_cspred(
    trajectory: str,
    topology: str,
    output: str,
    predictor: Tuple[str, ...] = ("UCBshift", "Sparta+"),
    pH: float = 7.0,
    tmpdir: str = "/tmp",
    overwrite: bool = False,
    sequence: str = "",
) -> None:
    """
    Predict chemical shifts for traj using UCBShift and/or Sparta+.
    Predictions are made frame by frame, and concatenated.
    Each frame is passed through PDBFixer to set the correct protonation state for given pH prior to being used as
    input for CS predictors
    """

    label = os.path.basename(trajectory).split(".")[0]
    if isinstance(predictor, str):
        predictor = (predictor,)
    if not overwrite:
        for p in predictor:
            outfile = os.path.join(output, f"{p}-{label}.csv")
            if os.path.exists(outfile):
                logger.info(f"'{outfile}' already exists, skipping it. use --overwrite to instead recompute")
                predictor = tuple(x for x in predictor if x != p)
        if len(predictor) == 0:
            return

    cspred_df = {}
    trj = md.load(trajectory, top=topology)
    if sequence:
        assert trj.top.to_fasta()[0] == sequence, (
            f"Amino acid sequence does not match: trj = {trj.top.to_fasta()[0]}, seq = {sequence}"
        )
    for i, frame in enumerate(tqdm(trj)):
        frame_file_i = os.path.join(tmpdir, f"{label}_{i}.pdb")
        frame.save(frame_file_i)
        if pH is not None:
            fixer = PDBFixer(filename=frame_file_i)
            fixer.addMissingHydrogens(pH=pH)
            with open(frame_file_i, "w") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
        tmp_trj = md.load(frame_file_i)
        for p in predictor:
            if p == "Sparta+":
                try:
                    if cspred_df.get(p) is None:
                        cspred_df[p] = md.chemical_shifts_spartaplus(tmp_trj)
                    else:
                        cspred_df[p] = pd.concat(
                            [cspred_df[p], md.chemical_shifts_spartaplus(tmp_trj).rename(columns={0: i})],
                            axis=1,
                        )
                except Exception as e:
                    logger.error(f"Error running {p} on frame {i}: {e}")
                    logger.error(traceback.format_exc())
            elif p == "UCBshift":
                try:
                    df = UCBshift.calc_sing_pdb(frame_file_i, pH, TP=False, ML=True, test=False)
                    df["frame"] = i
                    df = df.set_index(["RESNUM", "RESNAME", "frame"]).stack()
                    df.index.names = ["resSeq", None, "frame", "name"]
                    df = pd.pivot_table(df.to_frame(name="x"), values="x", index="frame", columns=["resSeq", "name"])
                    df.columns = pd.MultiIndex.from_tuples(
                        [(str(a[0]), a[1][:-2]) for a in df.columns], names=df.columns.names,
                    )
                    df = df.reset_index().drop(columns="frame").T
                    if cspred_df.get(p) is None:
                        cspred_df[p] = df
                    else:
                        cspred_df[p] = pd.concat([cspred_df[p], df.rename(columns={0: i})], axis=1)
                except Exception as e:
                    logger.error(f"Error running {p} on frame {i}: {e}")
                    logger.error(traceback.format_exc())
            else:
                raise ValueError(f"Predictor {p} not supported")
        os.remove(frame_file_i)
    for p in predictor:
        cspred_df[p].to_csv(os.path.join(output, f"{p}-{label}.csv"))


def main() -> None:
    run_cspred(**vars(get_args()))


if __name__ == "__main__":
    main()
