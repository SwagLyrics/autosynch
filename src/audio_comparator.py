import scipy.io.wavfile
import nussl
import commonfate
from mss_pytorch import mss_pytorch
from mss_pytorch.helpers import nnet_helpers

def ft2d(fpath):
    """Basic 2D Fourier transform vocal isolation method from nussl"""

    signal = nussl.AudioSignal(fpath)
    ft2d = nussl.FT2D(signal)
    ft2d.run()
    _, foreground = ft2d.make_audio_signals()

    return [(foreground.sample_rate, foreground.audio_data)]

def repet(fpath):
    """
    REPET original implementation.

    Z. Rafii and B. Pardo. "A Simple Music/Voice Separation Method based on the
    Extraction of the Repeating Musical Structure," 36th IEEE International
    Conference on Acoustics, Speech and Signal Processing, Prague, Czech
    Republic, May 22-27, 2011.
    """

    signal = nussl.AudioSignal(fpath)
    repet = nussl.Repet(signal)
    repet.run()
    _, foreground = repet.make_audio_signals()

    return [(foreground.sample_rate, foreground.audio_data)]

def repetsim(fpath):
    """
    REPET-SIM implementation.

    Z. Rafii and B. Pardo. "Music/Voice Separation using the Similarity Matrix,"
    13th International Society for Music Information Retrieval, Porto, Portugal,
    October 8-12, 2012.
    """

    signal = nussl.AudioSignal(fpath)
    repet_sim = nussl.RepetSim(signal)
    repet_sim.run()
    _, foreground = repet_sim.make_audio_signals()

    return [(foreground.sample_rate, foreground.audio_data)]

def hpss(fpath):
    """
    LibROSA Harmonic/Percussive Source Separation (HPSS) implementation.

    D. Fitzgerald. "Harmonic/percussive separation using median filtering," 13th
    International Conference on Digital Audio Effects, Graz, Austria, 2010.

    J. Driedger, M. Müller, S. Disch. "Extending harmonic-percussive separation
    of audio," 15th International Society for Music Information Retrieval
    Conference, Taipei, Taiwan, 2014.
    """

    signal = nussl.AudioSignal(fpath)
    hpss = nussl.HPSS(signal)
    hpss.run()
    _, foreground = hpss.make_audio_signals()

    return [(foreground.sample_rate, foreground.audio_data)]

def melodia(fpath):
    """
    Melodia implementation.

    Requires Melodia vamp plugin and vampy package:
    http://www.justinsalamon.com/news/melody-extraction-in-python-with-melodia

    J. Salamon and E. Gómez. "Melody Extraction from Polyphonic Music Signals
    Using Pitch Contour Characteristics," IEEE Transactions on Audio, Speech and
    Language Processing, 20(6):1759-1770, Aug. 2012.
    """

    signal = nussl.AudioSignal(fpath)
    melodia = nussl.Melodia(signal)
    melodia.run()
    _, foreground = melodia.make_audio_signals()

    return [(foreground.sample_rate, foreground.audio_data)]

def duet(fpath, num_sources=2, dominance=False):
    """
    DUET implementation.

    S. Rickard. "The DUET blind source separation algorithm," Blind Speech
    Separation. Springer Netherlands, 2007. 217-241.

    O. Yilmaz and S. Rickard. "Blind separation of speech mixtures via
    time-frequency masking," Signal Processing, IEEE Transactions on 52.7
    (2004): 1830-1847.

    :param fpath: Audio signal must be stereophonic.
    :param num_sources: Number of foreground sources to extract.
    :param dominance: Whether one source is more dominant than another
    :return sources: List of audio signals with length num_sources
    """

    if dominance:
        p = 0.5
    else:
        p = 1

    signal = nussl.AudioSignal(fpath)
    if not signal.is_stereo:
        return None

    duet = nussl.Duet(signal, num_sources, p=p)
    duet.run()
    sources = duet.make_audio_signals()

    return [(source.sample_rate, source.audio_data) for source in sources]

def rpca(fpath):
    """
    Robust Principal Component Analysis (RPCA) implementation.

    P.-S. Huang, S. D. Chen, P. Smaragdis, and M. Hasegawa-Johnson. "Singing-
    voice separation from monaural recordings using robust principal component
    analysis," Acoustics, Speech and Signal Processing, IEEE International
    Conference, 2012.
    """

    signal = nussl.AudioSignal(fpath)
    rpca = nussl.RPCA(signal)
    rpca.run()
    _, foreground = rpca.make_audio_signals()

    return [(foreground.sample_rate, foreground.audio_data)]

def msspytorch(fpath, training=False, apply_sparsity=True, olap=20, wsz=2049,
             N=4096, hop=384, B=16, mono=True):
    """
    MSS PyTorch implementation.

    S. I. Mimilakis, K. Drossos, J.F. Santos, G. Schuller, T. Virtanen, and Y.
    Bengio. "Monaural Singing Voice Separation with Skip-Filtering Connections
    and Recurrent Inference of Time-Frequency Mask," arXiv:1711.01437 [cs.SD],
    2017.
    """

    sfiltnet = mss_pytorch.main(training, apply_sparsity)
    sample_rate, signal = nnet_helpers.test_nnet(sfiltnet, fpath, olap, wsz, N, hop, B, mono)

    return [(sample_rate, signal)]

def madtwinnet(fpath):
    """
    Masker-Denoiser (MaD) TwinNet implementation.

    K. Drossos, S. I. Mimilakis, D. Serdyuk, G. Schuller, T. Virtanen, and Y.
    Bengio. "MaD TwinNet: Masker-Denoiser Architecture with Twin Networks for
    Monaural Sound Source Separation," arXiv:1802.00300 [cs.SD], 2018.
    """



def cfm(fpath, nb_components=2):
    """
    Common Fate Model/Transform implementation.

    F.-R. Stoter, A. Liutkus, R. Badeau, B. Edler, and P. Magron. "Common Fate
    Model for Unison Source Separation," 41st International Conference on
    Acoustics, Speech and Signal Processing, Shanghai, China, 2016.
    TITLE = {{Common Fate Model for Unison source Separation}},
    """
    import soundfile
    (signal, sample_rate) = soundfile.read(fpath, always_2d=True)
    components = commonfate.decompose.process(signal,
                    nb_components=nb_components, nb_iter=5)

    output = []
    for i in range(nb_components):
        output.append((sample_rate, components[i, ...]))

    return output
