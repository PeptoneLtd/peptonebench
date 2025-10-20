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

import numpy as np
from trizod.constants import AA1TO3, BBATNS
from trizod.potenci.potenci import getpredshifts
from trizod.scoring.scoring import compute_pscores, convert_to_triplet_data, get_offset_corrected_wSCS


def compute_gscores(
    cs: dict[tuple[int, str], float],  # {(res_num, atom_type): chemical_shift_value}
    sequence: str,
    temperature: float = 289.0,
    pH: float = 7.0,
    ionic_strength: float = 0.1,
) -> np.ndarray:
    """Compute g-scores from chemical shifts"""
    #### default parameters as 'unfiltered' ###
    offset_correction = True
    max_offset = np.inf
    reject_shift_type_only = True
    ###########################################

    shifts = []
    for (res, atom), val in cs.items():
        shifts.append(("None", "None", res, AA1TO3[sequence[res - 1]], atom, atom[0], val, 0.0, ""))
    random_coil_cs = getpredshifts(seq=sequence, temperature=temperature, pH=pH, ion=ionic_strength, pkacsvfile=False)
    ret = get_offset_corrected_wSCS(seq=sequence, shifts=shifts, predshiftdct=random_coil_cs)
    shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
    offsets = offf
    if not offset_correction:
        ashwi = ashwi0
        offsets = off0
    elif not (max_offset is None or np.isinf(max_offset)):
        for i, at in enumerate(BBATNS):
            if np.abs(offf[at]) > max_offset:
                offsets[at] = np.nan
                if reject_shift_type_only:
                    cmp_mask[:, i] = False
    if np.any(cmp_mask):
        ashwi3, k3 = convert_to_triplet_data(ashwi, cmp_mask)
        scores = compute_pscores(ashwi3, k3, cmp_mask)
    else:
        scores = np.full(len(sequence), np.nan)
    return scores
