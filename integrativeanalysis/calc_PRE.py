# requires DEER-PREdict: https://github.com/KULL-Centre/DEERpredict

import os
import shutil
import MDAnalysis
import numpy as np
from DEERPREdict.PRE import PREpredict
import pandas as pd
import pickle as pkl
import gzip

def find_fasta_filenames(path_to_dir, suffix=".fasta"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def find_dat_filenames(path_to_dir, suffix=".dat"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def find_pdb_filenames(path_to_dir, suffix=".pdb"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

# === CONFIG ===
models = ['alphafold','bioemu', 'boltz2', 'esmflow', 'esmfold', 'idpgan', 'idp-o', 'peptron_stable_pdb_idrome_20250812_256000', 'peptron_stable_pdb_20250809_236800']
fasta_dir = '../predictions/fasta/'
input_root = '../protonated_predictions/'  
exp_root = '../EXP_DATA/'
output_basename = 'calcPRE'
libname = 'MTSSL MMMx' # PRE rotamer library

# === Load protein list ===
files = find_fasta_filenames(fasta_dir)
proteins = [f[:-6] for f in files]
PREproteins = []

PREproteins_info = {}

for protein in proteins:
    datfiles = find_dat_filenames(exp_root+f"{protein}/")
    info = pd.read_csv(exp_root+f'{protein}/info.csv')
    PRE = 'no'
    for file in datfiles:
        if 'PRE' in file:
            PRE='yes'
            continue
    if PRE=='yes':
        PREproteins.append(protein)

        labelling_sites = []
        temps = []
        for file in datfiles:
            if 'PRE' in file:
                res = int(file.split('.')[0].split('-')[-1])
                dataset = file[:-4]
                temp = np.mean(info[info['Experiment']==dataset]['Temp(K)'])
                labelling_sites.append(res)
                temps.append(temp)

        PREproteins_info[protein] = {}
        PREproteins_info[protein]['Temperatures'] = temps
        PREproteins_info[protein]['Sites'] = labelling_sites
 
print(f"PRE proteins: {PREproteins}")

# === Iterate over models and proteins ===
for model in models:
    for protein in PREproteins:

        # iterate through all labelling sites
        for site, temp in zip(PREproteins_info[protein]['Sites'], PREproteins_info[protein]['Temperatures']):

            print(f"Processing {model}/{protein} PRE-label {site}...")

            if os.path.exists(input_root+f"{model}/{protein}/PREdata-{site}.npy"):
                print(f'Existing PRE calculations for {model}/{protein} and skipping...')
                continue
            else:
                os.makedirs(input_root+f"{model}/{protein}/{output_basename}", exist_ok=True)
                # check how many frames
                pdbfiles = find_pdb_filenames(input_root+f"{model}/{protein}/")
                nframes = len(pdbfiles)

                # calculate PREs for all frames, labelling sites
                for i in range(1, nframes+1):
                    PRE = PREpredict(MDAnalysis.Universe(input_root+f"{model}/{protein}/frame{i}.pdb"), residue = site, libname = libname,
                                     tau_t = .5*1e-9, log_file = input_root+f"{model}/{protein}/calcPRE/log", temperature = temp, z_cutoff = 0.05, atom_selection = 'H', Cbeta = False)
                    PRE.run(output_prefix = input_root+f"{model}/{protein}/calcPRE/PRE", tau_t = .5*1e-9,delay = 10e-3,
                            tau_c = 5*1e-09,r_2 = 10, wh = 750) # r_2 and wh don't matter yet here
                    
                    # save array
                    with gzip.open(input_root+f"{model}/{protein}/calcPRE/PRE-{site}.pkl", "rb") as f:
                        data = pkl.load(f)
                        tmpr3 = np.array(data['r3'])
                        tmpr6 = np.array(data['r6'])
                        tmpangular = np.array(data['angular'])
                    if i==1:
                        r3 = tmpr3
                        r6 = tmpr6
                        angular = tmpangular

                        # save residue data
                        d=np.loadtxt(input_root+f"{model}/{protein}/calcPRE/PRE-{site}.dat")
                        residues = d[:,0].astype(int)
                        vals = d[:,1].astype(float)
                        mask_nan = ~np.isnan(vals) # find valid residues (not terminal, labelling site or proline)
                        residues = residues[mask_nan]
                    else:
                        r3 = np.concatenate((r3, tmpr3),axis=0)
                        r6 = np.concatenate((r6, tmpr6),axis=0)
                        angular = np.concatenate((angular, tmpangular),axis=0)

                # save all frames together in an array and clean-up
                resdict = {}
                resdict['Residue'] = residues
                resdict['r3'] = r3
                resdict['r6'] = r6
                resdict['angular'] = angular
                np.save(input_root+f"{model}/{protein}/PREdata-{site}.npy", resdict)

                # remove tmp directory
                shutil.rmtree(input_root+f"{model}/{protein}/{output_basename}", ignore_errors=True)
            
print('All done!')