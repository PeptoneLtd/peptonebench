## code taken from https://github.com/microsoft/bioemu/blob/ac7455db1e981eec8267a20479dc274b857ed1c7/src/bioemu/convert_chemgraph.py#L296-L395

import mdtraj
import numpy as np


def _filter_unphysical_traj_masks(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    See `filter_unphysical_traj` for more details.
    """
    # CA-CA residue distance between sequential neighbouring pairs
    seq_contiguous_resid_pairs = np.array([(r.index, r.index + 1) for r in list(traj.topology.residues)[:-1]])

    ca_seq_distances, _ = mdtraj.compute_contacts(
        traj, scheme="ca", contacts=seq_contiguous_resid_pairs, periodic=False
    )
    ca_seq_distances = mdtraj.utils.in_units_of(ca_seq_distances, "nanometers", "angstrom")

    frames_match_ca_seq_distance = np.all(ca_seq_distances < max_ca_seq_distance, axis=1)

    # C-N distance between sequential neighbouring pairs
    cn_atom_pair_indices: list[tuple[int, int]] = []

    for resid_i, resid_j in seq_contiguous_resid_pairs:
        residue_i, residue_j = (
            traj.topology.residue(resid_i),
            traj.topology.residue(resid_j),
        )
        c_i, n_j = (
            list(residue_i.atoms_by_name("C")),
            list(residue_j.atoms_by_name("N")),
        )
        assert len(c_i) == len(n_j) == 1
        cn_atom_pair_indices.append((c_i[0].index, n_j[0].index))

    assert cn_atom_pair_indices

    cn_seq_distances = mdtraj.compute_distances(traj, cn_atom_pair_indices, periodic=False)
    cn_seq_distances = mdtraj.utils.in_units_of(cn_seq_distances, "nanometers", "angstrom")

    frames_match_cn_seq_distance = np.all(cn_seq_distances < max_cn_seq_distance, axis=1)

    # Clashes between any two atoms from different residues
    rest_distances, _ = mdtraj.compute_contacts(traj, periodic=False)
    frames_non_clash = np.all(
        mdtraj.utils.in_units_of(rest_distances, "nanometers", "angstrom") > clash_distance,
        axis=1,
    )
    return frames_match_ca_seq_distance, frames_match_cn_seq_distance, frames_non_clash


def _get_physical_traj_indices(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
    strict: bool = False,
) -> np.ndarray:
    """
    See `filter_unphysical_traj`. This returns trajectory frame indices satisfying certain physical criteria.
    """
    (
        frames_match_ca_seq_distance,
        frames_match_cn_seq_distance,
        frames_non_clash,
    ) = _filter_unphysical_traj_masks(traj, max_ca_seq_distance, max_cn_seq_distance, clash_distance)
    matches_all = frames_match_ca_seq_distance & frames_match_cn_seq_distance & frames_non_clash
    if strict:
        assert matches_all.sum() > 0, "Ended up with empty trajectory"
    return np.where(matches_all)[0]


def filter_unphysical_traj(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
    strict: bool = False,
) -> mdtraj.Trajectory:
    """
    Filters out 'unphysical' frames from a samples trajectory

    Args:
        traj: A trajectory object with multiple frames
        max_ca_seq_distance: Maximum carbon alpha distance between any two contiguous residues in the sequence (in Angstrom)
        max_cn_seq_distance: Maximum carbon-nitrogen distance between any two contiguous residues in the sequence (in Angstrom)
        clash_distance: Minimum distance between any two atoms belonging to different residues (in Angstrom)
        strict: Raises an error if all frames in `traj` are filtered out
    """
    matches_all = _get_physical_traj_indices(
        traj=traj,
        max_ca_seq_distance=max_ca_seq_distance,
        max_cn_seq_distance=max_cn_seq_distance,
        clash_distance=clash_distance,
        strict=strict,
    )
    return traj.slice(matches_all, copy=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter unphysical frames from a trajectory.")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to the trajectory file")
    parser.add_argument("--topology", type=str, default="", required=False, help="Path to the topology file, if needed")
    parser.add_argument("--outfile", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    print(f"Loading trajectory from {args.trajectory} with topology {args.topology if args.topology else 'None'}")
    trj = mdtraj.load(args.trajectory) if len(args.topology) == 0 else mdtraj.load(args.trajectory, top=args.topology)
    print(f"Loaded trajectory with {len(trj)} frames.")
    filtered_traj = filter_unphysical_traj(trj)
    print(f"Filtered trajectory has {len(filtered_traj)} frames.")
    filtered_traj.save(args.outfile)
