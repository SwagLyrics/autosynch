# autosynch repository

[![Build Status](https://travis-ci.com/chriswang030/autosynch.svg?branch=phase2)](https://travis-ci.com/chriswang030/autosynch) [![codecov](https://codecov.io/gh/chriswang030/autosynch/branch/phase2/graph/badge.svg)](https://codecov.io/gh/chriswang030/autosynch)

## installation
To install, do the following:
- `git clone` this branch (use `-b master --single-branch` to clone just this branch)
- `cd autosynch`
- `pip install -r requirements.txt`

## usage
To use, you must put a trained model into the `src/mad_twinnet/outputs/states`
directory. To download our weights trained on MedleyDB_V2 and run automatically,
simply execute:
- `./setup.sh` from the outer `autosynch` directory

If permission is denied, execute:
- `chmod a+rx setup.sh`
- `./setup.sh`

If you would like to download the weights manually or get a different version,
check here:
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3334973.svg)](https://doi.org/10.5281/zenodo.3334973) for weights trained on MedleyDB V1
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3351632.svg)](https://doi.org/10.5281/zenodo.3351632) for weights trained on MedleyDB V2

Currently, `align.py` gives an example run on The Avett Brothers' "Head Full of
Doubt/Road Full of Promise." A more detailed report on the output, as well as an
improved UI regarding alignment, will come later.
