# autosynch repository

[![Build Status](https://travis-ci.com/chriswang030/autosynch.svg?branch=phase2)](https://travis-ci.com/chriswang030/autosynch) [![codecov](https://codecov.io/gh/chriswang030/autosynch/branch/phase2/graph/badge.svg)](https://codecov.io/gh/chriswang030/autosynch)

WORK IN PROGRESS

This branch will contain additional work done in phase 2 of GSoC, including:
- hyphenation/syllable counting algorithms
- improvements to vocal isolation
- syllable alignment of music and lyrics
- initial testing for real-time processing

Get weights trained on MedleyDB_V1 here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3334973.svg)](https://doi.org/10.5281/zenodo.3334973)

Get weights trained on MedleyDB_V2 here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3351632.svg)](https://doi.org/10.5281/zenodo.3351632)

To automatically download the V2 weights to the right location, execute from the outer `autosynch` directory:
`curl 'https://zenodo.org/record/3351632/files/mad.pt?download=1' > autosynch/mad_twinnet/outputs/states/mad.pt`
