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

# code adapted from: https://github.com/paulrobustelli/Borthakur_MaxEnt_IDPs_2024/blob/main/calc_exp_data.py

import sys
import mdtraj as md
import os
import shutil
import math
import numpy as np
import textwrap
from Bio.PDB import *
from Bio.SeqUtils import seq1
import argparse

PALES_EXE = ### PATH TO PALES EXECUTABLE

# check python version
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# create parser
parser = argparse.ArgumentParser(prog='python calc_exp_data.py', \
         formatter_class=argparse.RawDescriptionHelpFormatter, \
         epilog=textwrap.dedent('''\
Required software/libraries:
- Python 3.x: https://www.python.org
- PALES: https://spin.niddk.nih.gov/bax/software/PALES
- mdtraj: http://mdtraj.org
- numpy: https://numpy.org
- pandas: https://pandas.pydata.org
- Biopython: https://biopython.org
 '''))
# define arguments
parser.add_argument('--directory', type=str, help='directory containing all pdb files')

args = parser.parse_args()

# find all pdb files in that directory
def find_pdb_filenames(path_to_dir, suffix=".pdb"):
  filenames = os.listdir(path_to_dir)
  return [filename for filename in filenames if filename.endswith(suffix)]

n_pdb_files = len(find_pdb_filenames(f"./{args.directory}/"))

# create a working directory labelled by ITASK_
wdir="rdcdir"+str(f"_{args.directory}")
os.makedirs(wdir, exist_ok=True)

# Define format for output
fmt0='%d,'; fmt1=''
for i in range(0, n_pdb_files-1): fmt1+='%.4lf,'
fmt1+='%.4lf'

# Calculate RDCs for all pdb files
# This requires PALES installed
# https://spin.niddk.nih.gov/bax/software/PALES/ 
if n_pdb_files>0:
 print("- Calculating RDC\n")
 for frame in range(1,n_pdb_files+1):
    # read trajectory files and topology (from pdb)
    # this requires mdtraj
    # http://mdtraj.org/1.9.3/
    trj = md.load(args.directory+f"/frame{frame}.pdb", top=args.directory+f"/frame{frame}.pdb")

    # create a temporary directory
    tmpdir=wdir+"/tmplocal-"+str(frame)
    os.makedirs(tmpdir, exist_ok=True)
    # save pdb file
    ipdb = tmpdir+"/out.pdb"
    trj[0].save_pdb(ipdb)
    # clean it - this requires Bio.PDB
    # https://biopython.org/wiki/Download
    structure = PDBParser().get_structure('PDB', ipdb)
    # get sequence info (residue name and number)
    # WARNING: assuming pdb with 1 model and 1 chain
    resname=[]; resnum=[] 
    for i in structure[0].get_chains():
     for j in i.get_residues(): 
         resname.append(j.get_resname())
         resnum.append(j.get_id()[1])
    # get sequence (one letter code)
    seq=seq1("".join(resname))
    # number of residues
    nres=len(resname)
    # sanity check of sequence length
    if(nres!=len(seq)):
      "Check length of the protein failed!"
      exit()
    # save clean pdb
    io = PDBIO()
    io.set_structure(structure)
    opdb = tmpdir+"/out-clean.pdb"
    io.save(opdb)
    # create PALES input file
    ifile = tmpdir+"/PALES_input.dat"
    f = open(ifile,"w")
    f.write("DATA SEQUENCE ")
    for i in range(0, nres):
        f.write("%s" % seq[i])
        # add a space every 10 residues, but not for last one
        if((i+1)%10==0 and i!=(nres-1)): f.write(" ")
    f.write("\n\n")
    f.write("VARS   RESID_I RESNAME_I ATOMNAME_I RESID_J RESNAME_J ATOMNAME_J D      DD    W\n")
    f.write("FORMAT %5d     %6s       %6s        %5d     %6s       %6s    %9.3f   %9.3f %.2f\n")
    f.write("\n")
    for i in range(0, nres):
        f.write("%d %3s H %d %3s N 0 1.00 1.00\n" % (resnum[i],resname[i],resnum[i],resname[i])) 
    f.close() 
    # run PALES on clean pdb
    rdc_frame=[]
    # cycle on residues, except first and last
    for ires in range(1, nres-1):
        # determine window
        l=7; h=7
        if(ires<7):      l=ires
        if(ires>nres-8): h=nres-1-ires
        # window is the minimum between l and h
        w=min(l,h)
        # output file
        ofile = tmpdir+"/"+str(resnum[ires])+".dat"
        # run PALES
        os.system(PALES_EXE+" -inD "+ifile+" -pdb "+opdb+" -r1 "+str(resnum[ires-w])+" -rN "+str(resnum[ires+w])+" -outD "+ofile)
        # parse the output file and add to list of RDCs 
        for lines in open(ofile, "r").readlines():
            riga=lines.strip().split()
            if(len(riga)==12 and riga[0].isdigit()):
              if(float(riga[0])==resnum[ires]): rdc_frame.append(float(riga[8]))
    # delete the temporary directory
    shutil.rmtree(tmpdir)
    # create/add to global numpy array (n_data, n_frames)
    if(frame==1):
      rdc=np.array(rdc_frame)
    else:
      rdc=np.column_stack((rdc,np.array(rdc_frame)))
 # save RDCs to file
 #label=np.array(resnum[1:nres-1])
 # Re-load structure using mdtraj for residue filtering
 pdb = trj
 # Get all residues excluding the first and last
 filtered = [
    (res.index, res.resSeq)
    for res in list(pdb.topology.residues)[1:-1]
    if res.name != "PRO"
    ]
 # Indices of residues to include in label and RDC
 valid_indices = [i for i, _ in filtered]
 label = np.array([resid for _, resid in filtered])
 np.savetxt(wdir+"/RDC.csv", np.column_stack((label,rdc)), fmt=fmt0+fmt1, header="resSeq,frame")

# closing log file
print('ALL DONE')
