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

from typing import List, Union, Dict
import os

import pandas as pd

def load_db(path: str) -> Dict[str, Dict[str, Union[str, float]]]:
    df = pd.read_csv(path).reset_index(drop=True).set_index("label")
    return df.to_dict(orient="index")


def list_pdbs(path: str) -> List[str]:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pdb")]