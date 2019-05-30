import numpy as np
from os.path import join
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.io.wavfile import read

import repet

mix = '/Users/Chris/autosynch/resources/MSD100/Mixtures/Dev'
src = '/Users/Chris/autosynch/resources/MSD100/Sources/Dev'

f1 = join(join(mix, 'BKS - Bulldozer'), 'mixture.wav')
f2 = join(join(src, 'BKS - Bulldozer'), 'vocals.wav')

sr1, y1 = read(f1)
sr2, y2 = read(f2)

bg = repet.sim(y1, sr1)
fg = y1 - bg

fg = np.nan_to_num(fg)
y2 = np.nan_to_num(y2)

dist, path = fastdtw(y2, y1, dist=euclidean)
print(dist)
