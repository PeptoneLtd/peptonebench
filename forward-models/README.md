# Forward models automation for PeptoneBench

## Building the docker image

1. download UCBShift / CSPred model weights from https://datadryad.org/stash/share/6vbrswTtNRcHk2vV3e6P1QGH1yYMhvdHDlauysTCObE
2. place the models.tgz file you downloaded at step 1) in the directory containing this readme file
3. run `docker build --platform=linux/amd64 . -t peptonebench-fwd-models`

## Running the forward models
First dowload all relevant ensembles and SAXS data to a directory structure like the following (SAXS data is not 
necessary if you only want to run Chemical shifts post-processing):

```aiignore
workdir/
 |- dataset.csv
 |- saxs_data/     # only needed for SAXS post-processing
 |- generators/
     |- bioemu/
     |- boltz1/
     |- boltz2/
  ...
```

then, for the SAXS forward model run
```aiignore
docker run -v $(pwd):/workdir --entrypoint /opt/conda/bin/python peptonebench-fwd-models \
    /home/mambauser/postprocess_saxs.py \
    --generators-dir /workdir/generators \
    --output-dir /workdir/processed_saxs  \
    --saxs-data /workdir/sasbdb-clean_data \
    --dataset /workdir/dataset.csv \
    --nprocs 8
```

or for the Chemical Shifts forward models run:
```aiignore
 docker run --platform=linux/amd64 -v $(pwd):/workdir --entrypoint /opt/conda/bin/python peptonebench-fwd-models \
    /home/mambauser/postprocess_nmr.py \
    --generators-dir /workdir/generators \
    --output-dir /workdir/processed_nmr  \
    --dataset /workdir/dataset.csv \
    --nprocs 8
```
**note**: UCBshift will fail if ran under rosetta on mac silicon