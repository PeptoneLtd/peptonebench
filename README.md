# PeptoneBench: evaluating protein ensembles across the order-disorder continuum

This repository contains all the code needed to run the the PeptoneBench benchmark to evaluate protein ensemble predictions against experimental data which covers structured as well as disordered proteins.
For details see the paper [Advancing Protein Ensemble Predictions Across the Order-Disorder Continuum](https://doi.org/10.1101/2025.10.18.680935). 
All datasets needed for the benchmarks can be found at https://doi.org/10.5281/zenodo.17306061.
The code related to the PepTron model can be found at https://github.com/peptoneltd/peptron.

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.10.18.680935-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.10.18.680935v2) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17306061.svg)](https://doi.org/10.5281/zenodo.17306061)

## Description
- `src/peptonebench`: main package to run reweighting and to obtain the benchmark scores. 
- `datasets`: description files for the three PeptoneDBs and a script to download the full datasets.
- `notebooks`: code to recreate the PeptoneDB-CS from the [BMRB](https://bmrb.io) and the PeptoneDB-SAXS from the [SASBDB](https://www.sasbdb.org), and code for generating figures of the paper.
- `forward-models`: code to process protein enmsebles to obtain the experimental chemical shifts values (CS) or scattering profiles (SAXS). Instructions and a separate Dockerfile are provided.
- `integrativeanalysis`: all the code and notebooks needed to perform the analysis on the PeptoneDB-Integrative dataset. A separate Dockerfile is provided.


## Instructions
Here are the steps needed to run PeptoneBench with a novel protein prediction model.
### 0. download experimental data
The benchmark is based on experimental data that should be downloaded into the `datasets` directory:
```
cd datasets
sh downloadDBs.sh
```
### 1. generate protein ensembles
The `datasets` directory contains the list of sequences in the three PeptoneDB datasets as .csv, for each entry generate an ensemble of configurations (at least 100).
### 2. forward model processing
Follow the instructions in the `forward-models` directory to prepare the ensembles and predict experimental observables with the proper forward model.
At the end you should have separate folders for each dataset, containing for each entry (LABEL) the generated ensembles (LABEL.xtc + LABEL.pdb) and the predicted observables (PREDICTOR-LABEL.csv).
### 3. running the benchmark
Here is an example of usage to analyse the ensemble predicted for the CS and the SAXS databases
```
PeptoneBench -f /path/to/mymodel/predictions_cs /path/to/mymodel/predictions_saxs
```
The script will automatically recognize from the name of the files which type of data they contain and compute the proper RMSE and reweighting.
As output you will have a file `rew_info-TAG.csv` containing the results of the benchmark with and without refinement, and `rew_weights-TAG.csv` containing the computed refinement weights for each protein conformation in the ensemble
### 4. exploring results
In the `noteboooks` directory we provide code for visualizing the benchmark results and reproducing the figures in the paper.
We provide code and notebooks in the `integrativeanalysis` direcory to analyse the entries of the PeptoneDB-Integrative dataset.


## Citation
```bibtex
@article{peptone2025,
  title     = {Advancing Protein Ensemble Predictions Across the Order-Disorder Continuum},
  author    = {Invernizzi, Michele and Bottaro, Sandro and Streit, Julian O and Trentini, Bruno and Venanzi, Niccolo AE and Reidenbach, Danny and Lee, Youhan and Dallago, Christian and Sirelkhatim, Hassan and Jing, Bowen and Airoldi, Fabio and Lindorff-Larsen, Kresten and Fisicaro, Carlo and Tamiola, Kamil},
  year      = 2025,
  journal   = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.1101/2025.10.18.680935},
  url       = {https://www.biorxiv.org/content/early/2025/10/19/2025.10.18.680935}
}
```

## License

Copyright 2025 Peptone Ltd

Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at:
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
