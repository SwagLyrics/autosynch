import math
import numpy as np
from scipy.io.wavfile import read
from os.path import join
import matplotlib.pyplot as plt


import repet

class VocalIsolator(object):
    def __init__(self, fpath):
        # Read in file and normalize
        self.sample_rate, self.audio_signal = read(fpath)
        self.audio_signal = self.audio_signal / np.sqrt(np.mean(self.audio_signal**2))
        self.audio_signal = self.audio_signal - np.mean(self.audio_signal)




# mix = '/Users/Chris/autosynch/resources/MSD100/Mixtures/Dev'
# src = '/Users/Chris/autosynch/resources/MSD100/Sources/Dev'
#
# f1 = join(join(mix, 'BKS - Bulldozer'), 'mixture.wav')
# f2 = join(join(src, 'BKS - Bulldozer'), 'vocals.wav')
#
# v = VocalIsolator(f1)
