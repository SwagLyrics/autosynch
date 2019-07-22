import os
import logging
from parselmouth import praat

from autosynch.config import praat_script_path

class SND(object):
    def __init__(self, silencedb=-25, mindip=2, minpause=0.3, showtext=2):
        """
        Syllable nuclei detector. Loads PRAAT script.

        :param silencedb: Threshold for maximum decibel to count as silence.
        :type silencedb: float
        :param mindip: Minimum dip in decibels to classify peak.
        :type mindip: float
        :param minpause: Minimum pause seconds to count as different syllable.
        :type minpause: float
        :param showtext: Flag to show text or not.
        :type showtext: int
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
        Runs Praat script. Returns None if file not found or file is not wav.

        :param file_path: Path to audio file to analyze.
        :type file_path: file-like
        :return: Timestamps of each syllable.
        :rtype: list[float] | None
        """

        # Check path validity
        if not os.path.exists(file_path):
            logging.error('{} does not exist'.format(file_path))
            return None
        if not os.path.splitext(file_path)[1] == '.wav':
            logging.error('{} is not a wav file'.format(file_path))
            return None

        script = self.script.format(silencedb=self.silencedb,
                                    mindip=self.mindip,
                                    minpause=self.minpause,
                                    showtext=self.showtext,
                                    file_path=file_path)
        output = praat.run(script, capture_output=True)[1].strip().split()

        return [float(time) for time in output[:-1]]
