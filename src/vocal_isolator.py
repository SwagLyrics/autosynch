
import numpy as np
import scipy.io.wavfile

import repet

class VocalIsolator(object):
    def __init__(self, fpath):
        self.sample_rate, audio_signal = scipy.io.wavfile.read(fpath)
        self.audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

    def isolate_center(self, fpath, fext, low_freq=150, high_freq=7000):
        # 150 to 7000 Hz typical range for human voice
        pass

    def repet_adaptive(self):
        background_signal = repet.adaptive(self.audio_signal, self.sample_rate)
        vocal_signal = audio_signal-background_signal

        return vocal_signal

    def repet_sim(self):
        background_signal = repet.sim(self.audio_signal, self.sample_rate)
        vocal_signal = audio_signal-background_signal

        return vocal_signal

    def virtanen(self):
        pass

    def fitzgerald(self):
        pass
