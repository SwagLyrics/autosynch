import os
import logging
import soundfile
import parselmouth

from config import resourcesdir

class SND(object):
    def __init__(self, silencedb=-25, mindip=2, minpause=0.3, showtext=2):
        """
        Syllable nuclei detector. Loads PRAAT script.

        :params: see praat-script.txt for keyword args description
        """

        self.silencedb = silencedb
        self.mindip = mindip
        self.minpause = minpause
        self.showtext = showtext

        # Run PRAAT script
        with open(os.path.join(resourcesdir, 'praat-script.txt'), 'r') as f:
            self.script = f.read()

    def run(self, fpath):
        """
        Runs PRAAT script.

        :param fpath: Path to audio file to analyze.
        :type fpath: str
        :return: Number of syllables
        :rtype: int
        """

        # Check path existence
        if not os.path.exists(fpath):
            logging.error('%s does not exist', fpath)
            raise FileNotFoundError('File does not exist')

        script = self.script.format(silencedb=self.silencedb,
                                    mindip=self.mindip,
                                    minpause=self.minpause,
                                    showtext=self.showtext,
                                    fpath=fpath)
        return parselmouth.praat.run(script, capture_output=True)[1].strip()
