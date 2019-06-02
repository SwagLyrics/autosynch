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


def ft2d(audio_signal):
    """Basic 2D Fourier transform vocal isolation method from nussl"""

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    ft2d = nussl.FT2D(signal)
    ft2d.run()
    _, foreground = ft2d.make_audio_signals()

    return foreground

def repet(audio_signal):
    """
    REPET original implementation.

    Z. Rafii and B. Pardo. "A Simple Music/Voice Separation Method based on the
    Extraction of the Repeating Musical Structure," 36th IEEE International
    Conference on Acoustics, Speech and Signal Processing, Prague, Czech
    Republic, May 22-27, 2011.
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    repet = nussl.Repet(signal)
    repet.run()
    _, foreground = repet.make_audio_signals()

    return foreground

def repet_sim(audio_signal):
    """
    REPET-SIM implementation.

    Z. Rafii and B. Pardo. "Music/Voice Separation using the Similarity Matrix,"
    13th International Society for Music Information Retrieval, Porto, Portugal,
    October 8-12, 2012.
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    repet_sim = nussl.RepetSim(signal)
    repet_sim.run()
    _, foreground = repet_sim.make_audio_signals()

    return foreground

def hpss(audio_signal):
    """
    LibROSA Harmonic/Percussive Source Separation (HPSS) implementation.

    D. Fitzgerald. "Harmonic/percussive separation using median filtering," 13th
    International Conference on Digital Audio Effects, Graz, Austria, 2010.

    J. Driedger, M. Müller, S. Disch. "Extending harmonic-percussive separation
    of audio," 15th International Society for Music Information Retrieval
    Conference, Taipei, Taiwan, 2014.
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    hpss = nussl.HPSS(signal)
    hpss.run()
    _, foreground = hpss.make_audio_signals()

    return foreground

def melodia(audio_signal):
    """
    Melodia implementation.

    Requires Melodia vamp plugin and vampy package:
    http://www.justinsalamon.com/news/melody-extraction-in-python-with-melodia

    J. Salamon and E. Gómez. "Melody Extraction from Polyphonic Music Signals
    Using Pitch Contour Characteristics," IEEE Transactions on Audio, Speech and
    Language Processing, 20(6):1759-1770, Aug. 2012.
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    melodia = nussl.Melodia(signal)
    melodia.run()
    _, foreground = melodia.make_audio_signals()

    return foreground

def duet(audio_signal, num_sources=1, dominance=False):
    """
    DUET implementation.

    S. Rickard. "The DUET blind source separation algorithm," Blind Speech
    Separation. Springer Netherlands, 2007. 217-241.

    O. Yilmaz and S. Rickard. "Blind separation of speech mixtures via
    time-frequency masking," Signal Processing, IEEE Transactions on 52.7
    (2004): 1830-1847.

    :param audio_signal: Audio signal must be stereophonic.
    :param num_sources: Number of foreground sources to extract.
    :param dominance: Whether one source is more dominant than another
    :return sources: List of audio signals with length num_sources
    """

    if not audio_signal.is_stereo():
        return None

    if dominance:
        p = 0.5
    else:
        p = 1

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    duet = nussl.Duet(signal, num_sources, p=p)
    duet.run()
    sources = duet.make_audio_signals()

    if num_sources == 1:
        return sources[0]
    else:
        return sources

def projet(audio_signal, num_sources=1):
    """
    PROJET implementation.

    D. Fitzgerald, A. Liutkus, and R. Badeau. "PROJET - Spatial Audio Separation
    Using Projections," 41st International Conference on Acoustics, Speech and
    Signal Processing, Shanghai, China, 2016.

    :param audio_signal: Audio signal must be stereophonic.
    :param num_sources: Number of foreground sources to extract.
    :return sources: List of audio signals with length num_sources
    """

    if not audio_signal.is_stereo():
        return None

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    projet = nussl.Projet(signal, num_sources)
    projet.run()
    sources = projet.make_audio_signals()

    if num_sources == 1:
        return sources[0]
    else:
        return sources

def nmf_mfcc(audio_signal, num_sources=1):
    """
    Non-Negative Matrix Factorization using K-Means Clustering on Mel-frequency
    Cepstral Coefficients (NMF MFCC) implementation.

    M. Spiertz and V. Gnann. "Source-filter based clustering for monaural blind
    source separation," Proceedings of the 12th International Conference on
    Digital Audio Effects, 2009.

    :param num_sources: Number of foreground sources to extract.
    :return sources: List of audio signals with length num_sources
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    nmf_mfcc = nussl.NMF_MFCC(signal, num_sources)
    nmf_mfcc.run()
    sources = nmf_mfcc.make_audio_signals()

    if num_sources == 1:
        return sources[0]
    else:
        return sources

def rpca(audio_signal):
    """
    Robust Principal Component Analysis (RPCA) implementation.

    P.-S. Huang, et al. "Singing-voice separation from monaural recordings using
    robust principal component analysis," Acoustics, Speech and Signal
    Processing, IEEE International Conference, 2012.
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    rpca = nussl.RPCA(signal)
    rpca.run()
    _, foreground = rpca.make_audio_signals()

    return foreground

def deep_clustering(audio_signal, num_sources=1):
    """
    Deep clustering implementation.

    J. R. Hershey, Z. Chen, J. Le Roux, and S. Watanabe. "Deep clustering:
    Discriminative embeddings for segmentation and separation," Acoustics,
    Speech and Signal Processing, IEEE International Conference, 2016.

    Y. Luo, Z. Chen, J. R. Hershey, J. Le Roux, and N. Mesgarani. "Deep
    Clustering and Conventional Networks for Music Separation: Stronger
    Together," arXiv, 2016.
    """

    signal = nussl.AudioSignal(audio_data_array=audio_signal)
    dc = nussl.DeepClustering(signal, num_sources=num_sources)
    dc.run()
    _, foreground = dc.make_audio_signals()

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
