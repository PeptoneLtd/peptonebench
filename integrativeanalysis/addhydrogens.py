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
import mdtraj as md
import numpy as np
import pandas as pd
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from io import StringIO

def find_fasta_filenames(path_to_dir, suffix=".fasta"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

# === CONFIG ===
# MAKE SURE TO REPLACE THESE PATHS WITH YOUR PATHS 

models = ['alphafold','bioemu', 'boltz2', 'esmflow', 'esmfold', 'idpgan', 'idp-o', 'peptron_stable_pdb_idrome_20250812_256000', 'peptron_stable_pdb_20250809_236800']
fasta_dir = # PATH TO FOLDER WITH ALL FASTA FILES
input_root = # PATH TO FOLDER WITH ALL HEAVY ATOM XTC/PDB FILES 
output_root = # OUTPUT ROOT PATH
exp_path = # PATH TO EXPERIMENTAL DATA 

# === Load protein list ===
files = find_fasta_filenames(fasta_dir)
proteins = [f[:-6] for f in files]
print(f"Proteins: {proteins}")

# === Load average pH values ===
pH_dict = {}
for prot in proteins:
    df = pd.read_csv(exp_path+f'{prot}/info.csv')
    pH_dict[prot] = np.mean(df['pH'])

# === Iterate over models and proteins ===
for model in models:
    for protein in proteins:
        input_path = os.path.join(input_root, model)
        input_model = os.path.join(input_path, f"{protein}.pdb")
        output_path = os.path.join(output_root, model, protein)

        if not os.path.exists(input_model):
            print(f"Skipping missing: {input_model}")
            continue

        if os.path.exists(output_path + "/frame1.pdb"):
            print(f"{input_model} already all frames protonated - continuing...")
            continue

        print(f"Processing {model}/{protein}...")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Load multi-frame PDB or XTC
        try:
            traj = md.load(input_path + f"/{protein}.xtc", top = input_path + f"/{protein}.pdb")
        except:
            traj = md.load(input_path + f"/{protein}.pdb", top = input_path + f"/{protein}.pdb")

        # Get average pH for this protein
        pH = pH_dict[protein]

        # Store fixed frames
        fixed_pdb_frames = []

        for i in range(traj.n_frames):
            print(f"Fixing frame {i+1}/{traj.n_frames}")
            temp_frame = f"temp_frame.pdb"
            traj[i].save_pdb(temp_frame)

            fixer = PDBFixer(filename=temp_frame)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(pH=pH)

            # Save fixed frame to string buffer
            f = StringIO()
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
            fixed_pdb_frames.append(f.getvalue())

        # Write all fixed frames into one multi-model PDB
        #with open(output_path + f"/{protein}.pdb", 'w') as out_pdb:
            #for frame_str in fixed_pdb_frames:
                #out_pdb.write(frame_str)
                #out_pdb.write("END\n")

        # Write each fixed frame as its own PDB file
        for i, frame_str in enumerate(fixed_pdb_frames):
            frame_filename = os.path.join(output_path, f"frame{i+1}.pdb")
            with open(frame_filename, 'w') as out_pdb:
                out_pdb.write(frame_str)
                out_pdb.write("END\n")

        print(f"Saved: {output_path}")

# Cleanup
if os.path.exists("temp_frame.pdb"):
    os.remove("temp_frame.pdb")
