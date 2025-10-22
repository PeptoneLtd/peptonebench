# Forward models automation for PeptoneBench

## Building the docker image
1. download UCBShift / CSPred model weights from https://datadryad.org/stash/share/6vbrswTtNRcHk2vV3e6P1QGH1yYMhvdHDlauysTCObE
2. place the models.tgz file you downloaded at step (1) in the directory containing this readme file
3. run `docker build --platform=linux/amd64 . -t peptonebench-fwd-models`

## Running forward model predictions
First generate all the all-atoms protein ensembles from the PeptoneDBs with your generavie model and save them in the following format:

```
workdir/
  |- mymodel-PeptoneDB-SAXS/
    |- SASDA25.pdb
    |- SASDA25.xtc
    |- SASDA37.pdb
    |- SASDA37.xtc
    ...
  |- mymodel-PeptoneDB-CS/
    |- 10036_1_1_1.pdb
    |- 10036_1_1_1.xtc
    |- 10091_1_1_1.pdb
    |- 10091_1_1_1.xtc
    ...
```
then run the forward models (it will take some time!) with
```
docker run -v $(pwd):/workdir peptonebench-fwd-models \
    -p Pepsi -f /workdir/mymodel-PeptoneDB-SAXS
docker run -v $(pwd):/workdir peptonebench-fwd-models \
    -p UCBshift -f /workdir/mymodel-PeptoneDB-CS
```
Finally one can run the benchmark script to reweight the ensembles and score them
```
PeptoneBench -f workdir/mymodel-PeptoneDB-SAXS workdir/mymodel-PeptoneDB-CS
```


## Reproducing the paper results
All the generated ensembles, with corresponding forward model predictions can be found at https://zenodo.org/records/17306061/files/Predictions.tar.gz.

To reproduce those results, first generate all protein ensembles following the instrucions presented in the [paper](https://doi.org/10.1101/2025.10.18.680935) for each generative model.
Setup a directory structure like the following:

```
workdir/
  |- PeptoneDB-SAXS/
    |- raw/
      |- bioemu/
      |- boltz1x/
      |- boltz2/
      ...
  |- PeptoneDB-CS/
    |- raw/
      |- bioemu/
      |- boltz1x/
      |- boltz2/
      ...
```

then, for the SAXS forward model run
```
docker run -v $(pwd):/workdir peptonebench-fwd-models \ 
    -p Pepsi \
    -f $(printf '/workdir/%s ' PeptoneDB-SAXS/raw/*) \
    --output-dir /workdir/PeptoneDB-SAXS/processed \
    --prepare-ensembles 
```

and for the CS forward models run:
```
docker run -v $(pwd):/workdir peptonebench-fwd-models \ 
    -p UCBshift_Sparta+ \
    -f $(printf '/workdir/%s ' PeptoneDB-CS/raw/*) \
    --output-dir /workdir/PeptoneDB-CS/processed \
    --prepare-ensembles  
```
**note**: UCBshift will fail if ran under rosetta on mac silicon

Finally run the benchmark on all the models
```
PeptoneBench -f workdir/PeptoneDB-SAXS/processed/* workdir/PeptoneDB-CS/processed/*
```
