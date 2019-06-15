import os
import logging
import soundfile
import parselmouth

from config import resourcesdir

def snd(fpath, silencedb=-25, mindip=2, minpause=0.3, showtext=2):
    """
    Runs PRAAT script to do syllable analysis on audio file.

    :param fpath: path of audio file
    :param **kwargs: see praat-script.txt for keyword args description
    :return: number of syllables
    """

    # Check path existence
    if not os.path.exists(fpath):
        logging.error('%s does not exist', fpath)
        raise FileNotFoundError('File does not exist')

    # Run PRAAT script
    with open(os.path.join(resourcesdir,'praat-script.txt'), 'r') as f:
        script = f.read()
    script = script.format(silencedb, mindip, minpause, showtext, fpath)
    syllables = parselmouth.praat.run(script, capture_output=True)[1].strip()

    return syllables
