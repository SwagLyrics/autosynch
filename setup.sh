#!/bin/sh

export PYTHONPATH=$PYTHONPATH:./
curl 'https://zenodo.org/record/3351632/files/mad.pt?download=1' > autosynch/mad_twinnet/outputs/states/mad.pt
python3 autosynch/align.py
