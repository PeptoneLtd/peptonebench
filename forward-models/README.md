# Forward models automation for PeptoneBench

## Building the docker image

1. download UCBShift / CSPred model weights from https://datadryad.org/stash/share/6vbrswTtNRcHk2vV3e6P1QGH1yYMhvdHDlauysTCObE
2. place the models.tgz file you downloaded at step 1) in this directory
3. run `docker build --platform=linux/amd64 . -t peptonebench-fwd-models`

## Running the forward models
First dowload all relevant ensembles and saxs data to a directory structure like the following:

```aiignore
workdir/
 |- dataset.csv
 |- saxs_data/
 |- ensembles/
     |- bioemu/
     |- boltz1/
     |- boltz2/
  ...
```

then, for the SAXS forward model run
```aiignore
docker run -v $(pwd):/workdir --entrypoint /opt/conda/bin/python peptonebench-fwd-models \
    /home/mambauser/postprocess_saxs.py \
    --generators bioemu,boltz1,boltz2 \
    --saxs-data /workdir/saxs_data \
    --dataset dataset.csv \
    --nprocs 8
```

or for the Chemical Shifts forward models run:
```aiignore
docker run -v $(pwd):/workdir --entrypoint /opt/conda/bin/python peptonebench-fwd-models \
    /home/mambauser/postprocess_nmr.py \
    --generators bioemu,boltz1,boltz2 \
    --dataset dataset.csv \
    --nprocs 8
```