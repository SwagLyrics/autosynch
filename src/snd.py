import os
import logging
import parselmouth as pm

from config import resourcesdir

def snd(fpath, silencedb=-25, mindip=2, minpause=0.3, showtext=2):
    """
    Runs PRAAT script to do syllable analysis on audio file.

    :param fpath: path of audio file
    :param **kwargs: see praat-script.txt for keyword args description
    :return: number of syllables
    """

    if not os.path.exists(fpath):
        logging.error('%s does not exist', fpath)

    with open(os.path.join(resourcesdir,'praat-script.txt'), 'r') as f:
        script = f.read()

    script = script.format(silencedb, mindip, minpause, showtext, fpath)

    return pm.praat.run(script, capture_output=True)[1].strip()
