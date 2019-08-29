#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./

# Download vocal isolation model
curl 'https://zenodo.org/record/3351632/files/mad.pt?download=1' > autosynch/mad_twinnet/outputs/states/mad.pt

# Install PortAudio
if [ "$(uname)" == "Darwin" ]; then
    brew install portaudio
    brew install sox
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    sudo apt-get update
    sudo apt-get install portaudio19-dev
    sudo apt-get install sox
fi
