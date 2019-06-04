import os
import parselmouth as pm

from config import resourcesdir

supported = ['.wav', '.aiff', '.aifc', '.au', '.mp3']

def run(fpath, silencedb=-25, mindip=2, minpause=0.3, showtext=2):
    """
    Runs PRAAT script to do syllable analysis on audio file.

    :param fpath: path of audio file
    :param **kwargs: see praat-script.txt for keyword args description
    :return: number of syllables
    """

    fext = os.path.splitext(fpath)[-1]
    if fext not in supported:
        print('{} extension not supported'.format(fext))

    with open(os.path.join(resourcesdir,'praat-script.txt'), 'r') as f:
        script = f.read()

    script = script.format(silencedb, mindip, minpause, showtext, fpath)

    return pm.praat.run(script, capture_output=True)[1].strip()
