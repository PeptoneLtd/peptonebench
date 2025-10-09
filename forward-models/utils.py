from typing import List, Union, Dict
import os

import pandas as pd

def load_db(path: str) -> Dict[str, Dict[str, Union[str, float]]]:
    df = pd.read_csv(path).reset_index(drop=True).set_index("label")
    return df.to_dict(orient="index")


def list_pdbs(path: str) -> List[str]:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pdb")]