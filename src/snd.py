import os
import logging
import soundfile
import parselmouth

from config import praat_script_path

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
        with open(praat_script_path, 'r') as f:
            self.script = f.read()

    def run(self, file_path):
        """
        Runs PRAAT script.

        :param file_path: Path to audio file to analyze.
        :type file_path: str
        :return: Number of syllables
        :rtype: int
        """

        # Check path existence
        if not os.path.exists(file_path):
            logging.error('%s does not exist', file_path)
            raise FileNotFoundError('File does not exist')

        script = self.script.format(silencedb=self.silencedb,
                                    mindip=self.mindip,
                                    minpause=self.minpause,
                                    showtext=self.showtext,
                                    file_path=file_path)
        return parselmouth.praat.run(script, capture_output=True)[1].strip()
