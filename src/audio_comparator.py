import numpy as np
import os

import config
from syllable_extractor import SyllableExtractor
import nussl

test_dir = os.path.join(config.resourcesdir, 'Unprocessed')
proc_dir = os.path.join(config.resourcesdir, 'Processed')

extractor = SyllableExtractor()

algorithms = []

for algorithm in algorithms:
    for file in os.listdir(test_dir):
        file = os.path.join(test_dir, file)


def ft2d(audio_signal, fname):
    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    ft2d = nussl.FT2D(signal)
    ft2d.run()
    _, foreground = ft2d.make_audio_signals()

    return foreground


# mix = '/Users/Chris/autosynch/resources/MSD100/Mixtures/Dev'
# src = '/Users/Chris/autosynch/resources/MSD100/Sources/Dev'
#
# f1 = join(join(mix, 'BKS - Bulldozer'), 'mixture.wav')
# f2 = join(join(src, 'BKS - Bulldozer'), 'vocals.wav')
#
# sr1, y1 = read(f1)
# sr2, y2 = read(f2)
#
# bg = repet.sim(y1, sr1)
# fg = y1 - bg
#
# fg = np.nan_to_num(fg)
# y2 = np.nan_to_num(y2)
#
# dist, path = fastdtw(y2, y1, dist=euclidean)
# print(dist)
