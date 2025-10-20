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
import os
import subprocess

import mdtraj as md
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pepsi-SAXS wrapper that can handle ensembles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectory", type=str, required=True, help="Trajectory file (e.g. .xtc)")
    parser.add_argument("--topology", type=str, required=True, help="Topology file (e.g. .pdb)")
    parser.add_argument(
        "--saxs",
        required=True,
        help="Path to experimental SAXS file, a .csv or .dat with columns [q, I(q), sigma]",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory for results")
    parser.add_argument(
        "--pepsi",
        default="Pepsi-SAXS",
        help="Path to the Pepsi-SAXS executable (default: assumes in PATH)",
    )
    parser.add_argument(
        "--pH",
        type=float,
        default=None,
        help="use PDBFixer to add hydrogens at given pH (default: do not add H)",
    )
    parser.add_argument(
        "--angular_units",
        type=int,
        default=None,
        help="Pepsi --angular_units option (default: automatic)",
    )
    parser.add_argument("--sequence", default="", help="Provide the aminoacid sequence to check for consistency")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary files (default: remove them)")

    return parser.parse_args()


def run_pepsi(
    trajectory: str,
    topology: str,
    saxs: str,
    output: str,
    pepsi: str = "Pepsi-SAXS",
    pH: float = None,
    angular_units: int = None,
    sequence: str = "",
    keep_tmp: bool = False,
) -> None:
    """
    Run Pepsi-SAXS on the provided trajectory and topology files.
    """

    flags = "--maximum_scattering_vector 10"
    if pH is not None:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer

        flags += " --hyd"
    if angular_units is not None:
        flags += f" --angular_units {angular_units}"  # 1 => 1/A, q = 4pi sin(theta)/lambda

    trj = md.load(trajectory, top=topology)
    if len(sequence) > 0:
        assert trj.top.to_fasta()[0] == sequence, (
            f"Amino acid sequence does not match: trj = {trj.top.to_fasta()[0]}, seq = {sequence}"
        )
    if saxs.endswith(".csv"):
        expt_df = pd.read_csv(saxs)
    elif saxs.endswith(".dat"):
        expt_df = pd.read_csv(saxs, sep="\s+", comment="#", names=["q", "I(q)", "sigma"])
    else:
        raise ValueError("SAXS file must be a .csv or .dat file.")
    os.makedirs(output, exist_ok=True)
    warning_log_file = os.path.join(output, "warning.log")
    if os.path.exists(warning_log_file):
        print(f"removing old warning log file, {warning_log_file}")
        os.remove(warning_log_file)
    saxs_tmp = os.path.join(output, os.path.basename(saxs)).replace(".csv", ".dat")
    expt_df.to_csv(saxs_tmp, sep="\t", index=False)

    frame_file = os.path.join(output, "frame_%04d.pdb")
    dat_file = frame_file.replace(".pdb", ".dat")
    log_file = dat_file.replace(".dat", ".log")

    log_df = []
    dat_df = []
    successful_frames = []
    for i, frame in enumerate(tqdm(trj)):
        frame_file_i = frame_file % i
        dat_file_i = dat_file % i
        log_file_i = log_file % i

        frame.save(frame_file_i)
        if pH is not None:
            fixer = PDBFixer(filename=frame_file_i)
            fixer.addMissingHydrogens(pH=pH)
            with open(frame_file_i, "w") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
        cmd = [pepsi, frame_file_i, saxs_tmp, "-o", dat_file_i] + flags.split()
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            try:
                high_n = 50
                warning_str = (
                    f"\n+++ WARNING pepsi-saxs failed for {trajectory} at frame {i}: {e} +++\n"
                    "+++ this could be due to a failure of the adaptive algorithm for selecting the expansion order,"
                    f" retrying with a fixed high number ({high_n}) of coefficients +++"
                )
                print(warning_str)
                with open(warning_log_file, "a") as warning_log:
                    warning_log.write(warning_str + "\n")
                with open(log_file_i, "w") as f:
                    subprocess.run(cmd + ["--nCoeff", str(high_n)], check=True, stdout=f, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                error_str = f"+++ ERROR processing {trajectory} at frame {i}: {e} +++"
                print(error_str)
                with open(warning_log_file, "a") as warning_log:
                    warning_log.write(error_str + "\n")
                continue

        # Check and save results
        all_dat_df = pd.read_csv(dat_file_i, sep="\s+", header=None, names=["q", "I(q)", "sigma", "I_fit"], comment="#")
        assert np.allclose(all_dat_df["q"], expt_df["q"], atol=1e-6), (
            f"q values do not match for frame {i}, {np.abs(all_dat_df['q'] - expt_df['q']).max():.2e}"
        )
        assert np.allclose(all_dat_df["I(q)"], expt_df["I(q)"], atol=1e-6), (
            f"I(q) values do not match for frame {i}, {np.abs(all_dat_df['I(q)'] - expt_df['I(q)']).max():.2e}"
        )
        assert np.allclose(all_dat_df["sigma"], expt_df["sigma"], atol=1e-6), (
            f"sigma values do not match for frame {i}, {np.abs(all_dat_df['sigma'] - expt_df['sigma']).max():.2e}"
        )
        dat_df.append(all_dat_df["I_fit"].to_numpy())
        successful_frames.append(i)

        # Save some log info
        d_rho = np.nan
        r0 = np.nan
        chi2 = np.nan
        displaced_volume = np.nan
        i0 = np.nan
        with open(log_file_i) as f:
            for line in f:
                if line.startswith("Best d_rho found"):
                    d_rho = float(line.split()[-2])
                elif line.startswith("Best r0 found"):
                    r0 = float(line.split()[-2])
                elif line.startswith("Chi^2"):
                    chi2 = float(line.split()[-1])
                elif line.startswith("Displaced Volume"):
                    displaced_volume = float(line.split()[-2])
                elif line.startswith("I(0)"):
                    i0 = float(line.split()[-1])
        log_df.append([d_rho, r0, chi2, displaced_volume, i0])

    # Save results to file
    label = os.path.basename(trajectory).split(".")[0]
    dat_df = pd.DataFrame(dat_df, columns=expt_df["q"].to_numpy(), index=successful_frames)
    dat_df.to_csv(os.path.join(output, f"Pepsi-{label}.csv"))
    log_df = pd.DataFrame(log_df, columns=["d_rho", "r0", "Chi^2", "displaced volume", "I(0)"], index=successful_frames)
    log_df.to_csv(os.path.join(output, f"Pepsi_log-{label}.csv"))

    # Clean up temporary files
    if not keep_tmp:
        os.remove(saxs_tmp)
        for i in successful_frames:
            os.remove(frame_file % i)
            os.remove(dat_file % i)
            os.remove(log_file % i)


if __name__ == "__main__":
    run_pepsi(**vars(get_args()))
