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

import os

DEFAULT_PEPTONEDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets"))
PEPTONEDB_PATH = os.getenv("PEPTONEDB_PATH", DEFAULT_PEPTONEDB_PATH)

DB_CS = os.path.join(PEPTONEDB_PATH, "PeptoneDB-CS/PeptoneDB-CS.csv")
DB_SAXS = os.path.join(PEPTONEDB_PATH, "PeptoneDB-SAXS/PeptoneDB-SAXS.csv")
DB_INTEGRATIVE = os.path.join(PEPTONEDB_PATH, "PeptoneDB-Integrative/PeptoneDB-Integrative.csv")

BMRB_DATA = os.path.join(PEPTONEDB_PATH, "PeptoneDB-CS/bmrb-data")
SASBDB_DATA = os.path.join(PEPTONEDB_PATH, "PeptoneDB-SAXS/sasbdb-clean_data")
INTEGRATIVE_DATA = os.path.join(PEPTONEDB_PATH, "PeptoneDB-Integrative")

BMRB_FILENAME = "bmrENTRYID_3.str"
SASBDB_FILENAME = "LABEL-bift.dat"
I_CS_FILENAME = "LABEL/CS.dat"
I_SAXS_FILENAME = "LABEL/SAXS_bift.dat"
GEN_FILENAME = "PREDICTOR-LABEL.csv"

DEFAULT_SAXS_PREDICTOR = "Pepsi"
DEFAULT_CS_PREDICTOR = "UCBshift"
DEFAULT_SELECTED_CS_TYPES = None
