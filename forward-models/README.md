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
docker run -v $(pwd):/workdir peptonebench-fwd-models \ 
    -p Pepsi \
    -f $(printf '/workdir/%s ' generators/*) \
    --output-dir /workdir/processed-saxs \
    --prepare-ensembles 
```
(omit `--prepare-ensembles` if the generators directories contain trajectories in pdb+xtc format)

or for the Chemical Shifts forward models run:
```aiignore
docker run -v $(pwd):/workdir peptonebench-fwd-models \ 
    -p UCBshift \
    -f $(printf '/workdir/%s ' generators/*) \
    --output-dir /workdir/processed-saxs \
    --prepare-ensembles  
```
**note**: UCBshift will fail if ran under rosetta on mac silicon