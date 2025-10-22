# Forward Models Automation for PeptoneBench

## Building the Docker Image
1. Download the UCBShift (CSPred) model weights from https://datadryad.org/stash/share/6vbrswTtNRcHk2vV3e6P1QGH1yYMhvdHDlauysTCObE
2. Place the `models.tgz` file you downloaded in step 1 into the directory containing this README file.
3. Build the Docker image by running:
  `docker build --platform=linux/amd64 . -t peptonebench-fwd-models`

## Running Forward Model Predictions
First, generate all-atom protein ensembles from the PeptoneDBs using your generative model, and save them in the following format:

```
workdir/
  ├─ mymodel-PeptoneDB-SAXS/
  │    ├─ SASDA25.pdb
  │    ├─ SASDA25.xtc
  │    ├─ SASDA37.pdb
  │    ├─ SASDA37.xtc
  │    ...
  ├─ mymodel-PeptoneDB-CS/
  │    ├─ 10036_1_1_1.pdb
  │    ├─ 10036_1_1_1.xtc
  │    ├─ 10091_1_1_1.pdb
  │    ├─ 10091_1_1_1.xtc
  │    ...
```

Next, run the forward models (this will take some time!):
```
docker run -v $(pwd):/workdir peptonebench-fwd-models \
    -p Pepsi -f /workdir/mymodel-PeptoneDB-SAXS
docker run -v $(pwd):/workdir peptonebench-fwd-models \
    -p UCBshift -f /workdir/mymodel-PeptoneDB-CS
```

Finally, run the benchmark script to reweight the ensembles and score them:
```
PeptoneBench -f workdir/mymodel-PeptoneDB-SAXS workdir/mymodel-PeptoneDB-CS
```



## Reproducing the Paper Results
All generated ensembles, along with corresponding forward model predictions, can be found at: https://zenodo.org/records/17306061/files/Predictions.tar.gz.

To reproduce these results, first generate all protein ensembles by following the instructions presented in the [paper](https://doi.org/10.1101/2025.10.18.680935) for each generative model.
Set up your directory structure as follows:

```
workdir/
  ├─ PeptoneDB-SAXS/
  │    └─ raw/
  │        ├─ bioemu/
  │        ├─ boltz1x/
  │        ├─ boltz2/
  │        ...
  ├─ PeptoneDB-CS/
  │    └─ raw/
  │        ├─ bioemu/
  │        ├─ boltz1x/
  │        ├─ boltz2/
  │        ...
```

Next, for the SAXS forward model, run:
```
docker run -v $(pwd):/workdir peptonebench-fwd-models \
    -p Pepsi \
    -f $(printf '/workdir/%s ' PeptoneDB-SAXS/raw/*) \
    --output-dir /workdir/PeptoneDB-SAXS/processed \
    --prepare-ensembles
```

For the CS forward models, run:
```
docker run -v $(pwd):/workdir peptonebench-fwd-models \
    -p UCBshift_Sparta+ \
    -f $(printf '/workdir/%s ' PeptoneDB-CS/raw/*) \
    --output-dir /workdir/PeptoneDB-CS/processed \
    --prepare-ensembles
```
**Note:** UCBshift will fail if run under Rosetta on Mac silicon.

Finally, run the benchmark on all models:
```
PeptoneBench -f workdir/PeptoneDB-SAXS/processed/* workdir/PeptoneDB-CS/processed/*
```
