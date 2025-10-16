# PeptoneBench: evaluating protein ensembles across the order-disorder continuum

This repository contains all the code needed to run the the PeptoneBench benchmark to evaluate protein ensemble predictions against experimental data which covers structured as well as disordered proteins.
For details see the preprint [Advancing Protein Ensemble Predictions Across the Order-Disorder Continuum](?).

The experimental data needed for the benchmark can be downloaded from Zenodo 
```
cd datasets
wget https://zenodo.org/api/records/17306061/files/PeptoneDBs.tar.gz
tar -xvzf PeptoneDBs.tar.gz
```

The directory `forward-models` contains the code to process generated protein enmsebles to obtain the experimental chemical shifts values (CS) or scattering profiles (SAXS).

The directory `peptonebench` contains the package to run reweighting and to obtain the benchmark scores. 
Here is an example of usage to analyse the ensemble predicted for the CS and the SAXS databases
```
PeptoneBench -f /path/to/mymodel/predictions_cs /path/to/mymodel/predictions_saxs
```
The script expects to find the typical output of the scripts in `forward-models`, thus for each enrty (LABEL) in the dataset the generated ensembles and the corresponding predicted observables as LABEL.pdb, LABEL.xtc, PREDICTOR-LABEL.csv
An example of this structure can be found in the Zenodo entry https://zenodo.org/uploads/17306061 as `Predictions.tar.gz`.

The directory `notebooks` contains the code to recreate the PeptoneDB-CS from the [BMRB](https://bmrb.io) and the PeptoneDB-SAXS from the [SASBDB](https://www.sasbdb.org), and code for generating most figures of the paper.

The directory `integrativeanalysis` contains all the code and notebooks needed to perform the in analysis on the PeptoneDB-Integrative dataset.
