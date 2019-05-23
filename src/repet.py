# Source: https://github.com/zafarrafii/REPET

"""
REpeating Pattern Extraction Technique (REPET) class
    Repetition is a fundamental element in generating and perceiving structure. In audio, mixtures are often composed of
    structures where a repeating background signal is superimposed with a varying foreground signal (e.g., a singer
    overlaying varying vocals on a repeating accompaniment or a varying speech signal mixed up with a repeating
    background noise). On this basis, we present the REpeating Pattern Extraction Technique (REPET), a simple approach
    for separating the repeating background from the non-repeating foreground in an audio mixture. The basic idea is to
    find the repeating elements in the mixture, derive the underlying repeating models, and extract the repeating
    background by comparing the models to the mixture. Unlike other separation approaches, REPET does not depend on
    special parameterizations, does not rely on complex frameworks, and does not require external information. Because
    it is only based on repetition, it has the advantage of being simple, fast, blind, and therefore completely and
    easily automatable.
Functions:
    original - original - REPET (original)
    extended - REPET extended
    adaptive - Adaptive REPET
    sim - REPET-SIM
    simonline - Online REPET-SIM
See also http://zafarrafii.com/#REPET
References:
    Zafar Rafii, Antoine Liutkus, and Bryan Pardo. "REPET for Background/Foreground Separation in Audio," Blind Source
    Separation, chapter 14, pages 395-411, Springer Berlin Heidelberg, 2014.
    Zafar Rafii and Bryan Pardo. "Online REPET-SIM for Real-time Speech Enhancement," 38th International Conference on
    Acoustics, Speech and Signal Processing, Vancouver, BC, Canada, May 26-31, 2013.
    Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US 20130064379 A1, March 2013.
    Zafar Rafii and Bryan Pardo. "REpeating Pattern Extraction Technique (REPET): A Simple Method for Music/Voice
    Separation," IEEE Transactions on Audio, Speech, and Language Processing, volume 21, number 1, pages 71-82,
    January, 2013.
    Zafar Rafii and Bryan Pardo. "Music/Voice Separation using the Similarity Matrix," 13th International Society on
    Music Information Retrieval, Porto, Portugal, October 8-12, 2012.
    Antoine Liutkus, Zafar Rafii, Roland Badeau, Bryan Pardo, and Gaël Richard. "Adaptive Filtering for Music/Voice
    Separation Exploiting the Repeating Musical Structure," 37th International Conference on Acoustics, Speech and
    Signal Processing, Kyoto, Japan, March 25-30, 2012.
    Zafar Rafii and Bryan Pardo. "A Simple Music/Voice Separation Method based on the Extraction of the Repeating
    Musical Structure," 36th International Conference on Acoustics, Speech and Signal Processing, Prague, Czech
    Republic, May 22-27, 2011.
Author:
    Zafar Rafii
    zafarrafii@gmail.com
    http://zafarrafii.com
    https://github.com/zafarrafii
    https://www.linkedin.com/in/zafarrafii/
    07/12/18
"""

import numpy as np
import scipy.signal


# Public functions/variables
# Window length in samples (audio stationary around 40 ms; power of 2 for fast FFT and constant overlap-add)
windowlength = lambda sample_rate: 2**int(np.ceil(np.log2(0.04*sample_rate)))

# Window function (periodic Hamming window for constant overlap-add)
windowfunction = lambda window_length: scipy.signal.hamming(window_length, False)

# Step length (half the window length for constant overlap-add)
steplength = lambda window_length: round(window_length/2)

# Cutoff frequency in Hz for the dual high-pass filter of the foreground (vocals are rarely below 100 Hz)
cutoff_frequency = 100

# Period range in seconds for the beat spectrum (for REPET, REPET extented, and adaptive REPET)
period_range = np.array([1, 10])

# Segment length and step in seconds (for REPET extented and adaptive REPET)
segment_length = 10
segment_step = 5

# Filter order for the median filter (for adaptive REPET)
filter_order = 5

# Minimal threshold for two similar frames in [0,1], minimal distance between two similar frames in seconds, and maximal
# number of similar frames for one frame (for REPET-SIM and online REPET-SIM)
similarity_threshold = 0
similarity_distance = 1
similarity_number = 100

# Buffer length in seconds (for online REPET-SIM)
buffer_length = 10


# Public functions
def original(audio_signal, sample_rate):
    """
    repet REPET (original)
        The original REPET aims at identifying and extracting the repeating patterns in an audio mixture, by estimating
        a period of the underlying repeating structure and modeling a segment of the periodically repeating background.
        background_signal = repet.original(audio_signal, sample_rate)
    Arguments:
        audio_signal: audio signal [number_samples, number_channels]
        sample_rate: sample rate in Hz
        background_signal: background signal [number_samples, number_channels]
    Example: Estimate the background and foreground signals, and display their spectrograms
        # Import modules
        import scipy.io.wavfile
        import repet
        import numpy as np
        import matplotlib.pyplot as plt
        # Audio signal (normalized) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        # Estimate the background signal and infer the foreground signal
        background_signal = repet.original(audio_signal, sample_rate);
        foreground_signal = audio_signal-background_signal;
        # Write the background and foreground signals (un-normalized)
        scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
        scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)
        # Compute the audio, background, and foreground spectrograms
        window_length = repet.windowlength(sample_rate)
        window_function = repet.windowfunction(window_length)
        step_length = repet.steplength(window_length)
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        # Display the audio, background, and foreground spectrograms (up to 5kHz)
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1)
        plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Audio Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 2)
        plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Background Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 3)
        plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Foreground Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Window length, window function, and step length for the STFT
    window_length = windowlength(sample_rate)
    window_function = windowfunction(window_length)
    step_length = steplength(window_length)

    # Number of time frames
    number_times = int(np.ceil((window_length-step_length+number_samples)/step_length))

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # STFT of the current channel
        audio_stft[:, :, channel_index] = _stft(audio_signal[:, channel_index], window_function, step_length)

    # Magnitude spectrogram (with DC component and without mirrored frequencies)
    audio_spectrogram = abs(audio_stft[0:int(window_length/2)+1, :, :])

    # Beat spectrum of the spectrograms averaged over the channels (squared to emphasize peaks of periodicitiy)
    beat_spectrum = _beatspectrum(np.power(np.mean(audio_spectrogram, axis=2), 2))

    # Period range in time frames for the beat spectrum
    period_range2 = np.round(period_range*sample_rate/step_length).astype(int)

    # Repeating period in time frames given the period range
    repeating_period = _periods(beat_spectrum, period_range2)

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(np.ceil(cutoff_frequency*(window_length-1)/sample_rate))-1

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # Repeating mask for the current channel
        repeating_mask = _mask(audio_spectrogram[:, :, channel_index], repeating_period)

        # High-pass filtering of the dual foreground
        repeating_mask[1:cutoff_frequency2+2, :] = 1

        # Mirror the frequency channels
        repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1, :]))

        # Estimated repeating background for the current channel
        background_signal1 = _istft(repeating_mask*audio_stft[:, :, channel_index], window_function, step_length)

        # Truncate to the original number of samples
        background_signal[:, channel_index] = background_signal1[0:number_samples]

    return background_signal


def extended(audio_signal, sample_rate):
    """
    extended REPET extended
        The original REPET can be easily extended to handle varying repeating structures, by simply applying the method
        along time, on individual segments or via a sliding window.
        background_signal = repet.extended(audio_signal, sample_rate)
    Arguments:
        audio_signal: audio signal [number_samples, number_channels]
        sample_rate: sample rate in Hz
        background_signal: background signal [number_samples, number_channels]
    Example: Estimate the background and foreground signals, and display their spectrograms
        # Import modules
        import scipy.io.wavfile
        import repet
        import numpy as np
        import matplotlib.pyplot as plt
        # Audio signal (normalized) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        # Estimate the background signal and infer the foreground signal
        background_signal = repet.extended(audio_signal, sample_rate);
        foreground_signal = audio_signal-background_signal;
        # Write the background and foreground signals (un-normalized)
        scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
        scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)
        # Compute the audio, background, and foreground spectrograms
        window_length = repet.windowlength(sample_rate)
        window_function = repet.windowfunction(window_length)
        step_length = repet.steplength(window_length)
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        # Display the audio, background, and foreground spectrograms (up to 5kHz)
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1)
        plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Audio Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 2)
        plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Background Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 3)
        plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Foreground Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Segmentation length, step, and overlap in samples
    segment_length2 = round(segment_length*sample_rate)
    segment_step2 = round(segment_step*sample_rate)
    segment_overlap2 = segment_length2-segment_step2

    # One segment if the signal is too short
    if number_samples < segment_length2+segment_step2:
        number_segments = 1
    else:

        # Number of segments (the last one could be longer)
        number_segments = 1+int(np.floor((number_samples-segment_length2)/segment_step2))

        # Triangular window for the overlapping parts
        segment_window = scipy.signal.triang(2*segment_overlap2)

    # Window length, window function, and step length for the STFT
    window_length = windowlength(sample_rate)
    window_function = windowfunction(window_length)
    step_length = steplength(window_length)

    # Period range in time frames for the beat spectrum
    period_range2 = np.round(period_range*sample_rate/step_length).astype(int)

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(np.ceil(cutoff_frequency*(window_length-1)/sample_rate))-1

    # Initialize background signal
    background_signal = np.zeros((number_samples,number_channels))

    # Loop over the segments
    for segment_index in range(0, number_segments):

        # Case one segment
        if number_segments == 1:
            audio_segment = audio_signal
            segment_length2 = number_samples
        else:

            # Sample index for the segment
            sample_index = segment_index*segment_step2

            # Case first segments (same length)
            if segment_index < number_segments-1:
                audio_segment = audio_signal[sample_index:sample_index+segment_length2, :]

            # Case last segment (could be longer)
            elif segment_index == number_segments-1:
                audio_segment = audio_signal[sample_index:number_samples, :]
                segment_length2 = len(audio_segment)

        # Number of time frames
        number_times = int(np.ceil((window_length-step_length+segment_length2)/step_length))

        # Initialize the STFT
        audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # STFT of the current channel
            audio_stft[:, :, channel_index] = _stft(audio_segment[:, channel_index], window_function, step_length)

        # Magnitude spectrogram (with DC component and without mirrored frequencies)
        audio_spectrogram = abs(audio_stft[0:int(window_length/2)+1, :, :])

        # Beat spectrum of the spectrograms averaged over the channels (squared to emphasize peaks of periodicitiy)
        beat_spectrum = _beatspectrum(np.power(np.mean(audio_spectrogram, axis=2), 2))

        # Initialize the background signal
        background_segment = np.zeros((segment_length2, number_channels))

        # Repeating period in time frames given the period range
        repeating_period = _periods(beat_spectrum, period_range2)

        # Initialize the background segment
        background_segment = np.zeros((segment_length2, number_channels))

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Repeating mask for the current channel
            repeating_mask = _mask(audio_spectrogram[:, :, channel_index], repeating_period)

            # High-pass filtering of the dual foreground
            repeating_mask[1:cutoff_frequency2+2, :] = 1

            # Mirror the frequency channels
            repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1, :]))

            # Estimated repeating background for the current channel
            background_segment1 = _istft(repeating_mask*audio_stft[:, :, channel_index], window_function, step_length)

            # Truncate to the original number of samples
            background_segment[:, channel_index] = background_segment1[0:segment_length2]

        # Case one segment
        if number_segments == 1:
            background_signal = background_segment
        else:

            # Case first segment
            if segment_index == 0:
                background_signal[0:segment_length2, :] = background_signal[0:segment_length2, :] + background_segment

            # Case last segments
            elif segment_index <= number_segments-1:

                # Half windowing of the overlap part of the background signal on the right
                background_signal[sample_index:sample_index+segment_overlap2, :] \
                    = background_signal[sample_index:sample_index+segment_overlap2, :]\
                      *segment_window[segment_overlap2:2*segment_overlap2, np.newaxis]

                # Half windowing of the overlap part of the background segment on the left
                background_segment[0:segment_overlap2, :] \
                    = background_segment[0:segment_overlap2, :]*segment_window[0:segment_overlap2, np.newaxis]
                background_signal[sample_index:sample_index+segment_length2, :] \
                    = background_signal[sample_index:sample_index+segment_length2, :] + background_segment

    return background_signal


def adaptive(audio_signal, sample_rate):
    """
    adaptive Adaptive REPET
        The original REPET works well when the repeating background is relatively stable (e.g., a verse or the chorus in
        a song); however, the repeating background can also vary over time (e.g., a verse followed by the chorus in the
        song). The adaptive REPET is an extension of the original REPET that can handle varying repeating structures, by
        estimating the time-varying repeating periods and extracting the repeating background locally, without the need
        for segmentation or windowing.
        background_signal = repet.adaptive(audio_signal, sample_rate)
    Arguments:
        audio_signal: audio signal [number_samples, number_channels]
        sample_rate: sample rate in Hz
        background_signal: background signal [number_samples, number_channels]
    Example: Estimate the background and foreground signals, and display their spectrograms
        # Import modules
        import scipy.io.wavfile
        import repet
        import numpy as np
        import matplotlib.pyplot as plt
        # Audio signal (normalized) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        # Estimate the background signal and infer the foreground signal
        background_signal = repet.adaptive(audio_signal, sample_rate);
        foreground_signal = audio_signal-background_signal;
        # Write the background and foreground signals (un-normalized)
        scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
        scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)
        # Compute the audio, background, and foreground spectrograms
        window_length = repet.windowlength(sample_rate)
        window_function = repet.windowfunction(window_length)
        step_length = repet.steplength(window_length)
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        # Display the audio, background, and foreground spectrograms (up to 5kHz)
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1)
        plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Audio Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 2)
        plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Background Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 3)
        plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Foreground Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Window length, window function, and step length for the STFT
    window_length = windowlength(sample_rate)
    window_function = windowfunction(window_length)
    step_length = steplength(window_length)

    # Number of time frames
    number_times = int(np.ceil((window_length-step_length+number_samples)/step_length))

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # STFT of the current channel
        audio_stft[:, :, channel_index] = _stft(audio_signal[:, channel_index], window_function, step_length)

    # Magnitude spectrogram (with DC component and without mirrored frequencies)
    audio_spectrogram = abs(audio_stft[0:int(window_length/2)+1, :, :])

    # Segment length and step in time frames for the beat spectrogram
    segment_length2 = int(round(segment_length*sample_rate/step_length))
    segment_step2 = int(round(segment_step * sample_rate / step_length))

    # Beat spectrogram of the spectrograms averaged over the channels (squared to emphasize peaks of periodicitiy)
    beat_spectrogram = _beatspectrogram(np.power(np.mean(audio_spectrogram, axis=2), 2), segment_length2, segment_step2)

    # Period range in time frames for the beat spectrogram
    period_range2 = np.round(period_range*sample_rate/step_length).astype(int)

    # Repeating periods in time frames given the period range
    repeating_periods = _periods(beat_spectrogram, period_range2)

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(np.ceil(cutoff_frequency*(window_length-1)/sample_rate))-1

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # Repeating mask for the current channel
        repeating_mask = _adaptivemask(audio_spectrogram[:, :, channel_index], repeating_periods, filter_order)

        # High-pass filtering of the dual foreground
        repeating_mask[1:cutoff_frequency2+2, :] = 1

        # Mirror the frequency channels
        repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1, :]))

        # Estimated repeating background for the current channel
        background_signal1 = _istft(repeating_mask*audio_stft[:, :, channel_index], window_function, step_length)

        # Truncate to the original number of samples
        background_signal[:, channel_index] = background_signal1[0:number_samples]

    return background_signal


def sim(audio_signal, sample_rate):
    """
    sim REPET-SIM
        The REPET methods work well when the repeating background has periodically repeating patterns (e.g., jackhammer
        noise); however, the repeating patterns can also happen intermittently or without a global or local periodicity
        (e.g., frogs by a pond). REPET-SIM is a generalization of REPET that can also handle non-periodically repeating
        structures, by using a similarity matrix to identify the repeating elements.
        background_signal = repet.sim(audio_signal, sample_rate)
    Arguments:
        audio_signal: audio signal [number_samples, number_channels]
        sample_rate: sample rate in Hz
        background_signal: background signal [number_samples, number_channels]
    Example: Estimate the background and foreground signals, and display their spectrograms
        # Import modules
        import scipy.io.wavfile
        import repet
        import numpy as np
        import matplotlib.pyplot as plt
        # Audio signal (normalized) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        # Estimate the background signal and infer the foreground signal
        background_signal = repet.sim(audio_signal, sample_rate);
        foreground_signal = audio_signal-background_signal;
        # Write the background and foreground signals (un-normalized)
        scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
        scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)
        # Compute the audio, background, and foreground spectrograms
        window_length = repet.windowlength(sample_rate)
        window_function = repet.windowfunction(window_length)
        step_length = repet.steplength(window_length)
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        # Display the audio, background, and foreground spectrograms (up to 5kHz)
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1)
        plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Audio Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 2)
        plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Background Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 3)
        plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Foreground Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Window length, window function, and step length for the STFT
    window_length = windowlength(sample_rate)
    window_function = windowfunction(window_length)
    step_length = steplength(window_length)

    # Number of time frames
    number_times = int(np.ceil((window_length-step_length+number_samples)/step_length))

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # STFT of the current channel
        audio_stft[:, :, channel_index] = _stft(audio_signal[:, channel_index], window_function, step_length)

    # Magnitude spectrogram (with DC component and without mirrored frequencies)
    audio_spectrogram = abs(audio_stft[0:int(window_length/2)+1, :, :])

    # Self-similarity of the spectrograms averaged over the channels
    similarity_matrix = _selfsimilaritymatrix(np.mean(audio_spectrogram, axis=2))

    # Similarity distance in time frames
    similarity_distance2 = int(round(similarity_distance*sample_rate/step_length))

    # Similarity indices for all the frames
    similarity_indices = _indices(similarity_matrix, similarity_threshold, similarity_distance2, similarity_number)

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(np.ceil(cutoff_frequency*(window_length-1)/sample_rate))-1

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # Repeating mask for the current channel
        repeating_mask = _simmask(audio_spectrogram[:, :, channel_index], similarity_indices)

        # High-pass filtering of the dual foreground
        repeating_mask[1:cutoff_frequency2+2, :] = 1

        # Mirror the frequency channels
        repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1, :]))

        # Estimated repeating background for the current channel
        background_signal1 = _istft(repeating_mask*audio_stft[:, :, channel_index], window_function, step_length)

        # Truncate to the original number of samples
        background_signal[:, channel_index] = background_signal1[0:number_samples]

    return background_signal


def simonline(audio_signal, sample_rate):
    """
    simonline Online REPET-SIM
        REPET-SIM can be easily implemented online to handle real-time computing, particularly for real-time speech
        enhancement. The online REPET-SIM simply processes the time frames of the mixture one after the other given a
        buffer that temporally stores past frames.
        background_signal = repet.sim(audio_signal, sample_rate)
    Arguments:
        audio_signal: audio signal [number_samples, number_channels]
        sample_rate: sample rate in Hz
        background_signal: background signal [number_samples, number_channels]
    Example: Estimate the background and foreground signals, and display their spectrograms
        # Import modules
        import scipy.io.wavfile
        import repet
        import numpy as np
        import matplotlib.pyplot as plt
        # Audio signal (normalized) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        # Estimate the background signal and infer the foreground signal
        background_signal = repet.simonline(audio_signal, sample_rate);
        foreground_signal = audio_signal-background_signal;
        # Write the background and foreground signals (un-normalized)
        scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
        scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)
        # Compute the audio, background, and foreground spectrograms
        window_length = repet.windowlength(sample_rate)
        window_function = repet.windowfunction(window_length)
        step_length = repet.steplength(window_length)
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
        # Display the audio, background, and foreground spectrograms (up to 5kHz)
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1)
        plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Audio Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 2)
        plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Background Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.subplot(3, 1, 3)
        plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
        plt.title('Foreground Spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/8*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Window length, window function, and step length for the STFT
    window_length = windowlength(sample_rate)
    window_function = windowfunction(window_length)
    step_length = steplength(window_length)

    # Number of time frames
    number_times = int(np.ceil((number_samples-window_length)/step_length+1))

    # Buffer length in time frames
    buffer_length2 = int(round((buffer_length*sample_rate-window_length)/step_length+1))

    # Initialize the buffer spectrogram
    buffer_spectrogram = np.zeros((int(window_length/2+1), buffer_length2, number_channels))

    # Loop over the time frames to compute the buffer spectrogram (the last frame will be the frame to be processed)
    for time_index in range(0, buffer_length2-1):

        # Sample index in the signal
        sample_index = step_length*time_index

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Compute the FT of the segment
            buffer_ft = np.fft.fft(audio_signal[sample_index:window_length+sample_index, channel_index]
                                   * window_function, axis=0)

            # Derive the spectrum of the frame
            buffer_spectrogram[:, time_index, channel_index] = abs(buffer_ft[0:int(window_length/2+1)])

    # Zero-pad the audio signal at the end
    audio_signal = np.pad(audio_signal, (0, (number_times-1)*step_length+window_length-number_samples),
                          'constant', constant_values=0)

    # Similarity distance in time frames
    similarity_distance2 = int(round(similarity_distance*sample_rate/step_length))

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(np.ceil(cutoff_frequency*(window_length-1)/sample_rate))-1

    # Initialize the background signal
    background_signal = np.zeros(((number_times-1)*step_length+window_length, number_channels))

    # Loop over the time frames to compute the background signal
    for time_index in range(buffer_length2-1, number_times):

        # Sample index in the signal
        sample_index = step_length*time_index

        # Time index of the current frame
        current_index = time_index % buffer_length2

        # Initialize the FT of the current segment
        current_ft = np.zeros((window_length, number_channels), dtype=complex)

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Compute the FT of the current segment
            current_ft[:, channel_index] = np.fft.fft(audio_signal[sample_index:window_length+sample_index,
                                                      channel_index]*window_function, axis=0)

            # Derive the spectrum of the current frame and update the buffer spectrogram
            buffer_spectrogram[:, current_index, channel_index] \
                = np.abs(current_ft[0:int(window_length/2+1), channel_index])

        # Cosine similarity between the spectrum of the current frame and the past frames, for all the channels
        similarity_vector = _similaritymatrix(np.mean(buffer_spectrogram, axis=2),
                                              np.mean(buffer_spectrogram[:, current_index:current_index+1, :], axis=2))

        # Indices of the similar frames
        _, similarity_indices = _localmaxima(similarity_vector[:, 0], similarity_threshold, similarity_distance2,
                                             similarity_number)

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Compute the repeating spectrum for the current frame
            repeating_spectrum = np.median(buffer_spectrogram[:, similarity_indices, channel_index], axis=1)

            # Refine the repeating spectrum
            repeating_spectrum = np.minimum(repeating_spectrum,
                                            buffer_spectrogram[:, current_index, channel_index])

            # Derive the repeating mask for the current frame
            repeating_mask = (repeating_spectrum+np.finfo(float).eps)\
            / (buffer_spectrogram[:, current_index, channel_index]+np.finfo(float).eps)

            # High-pass filtering of the dual foreground
            repeating_mask[1:cutoff_frequency2+2] = 1

            # Mirror the frequency channels
            repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1]))

            # Apply the mask to the FT of the current segment
            background_ft = repeating_mask*current_ft[:, channel_index]

            # Inverse FT of the current segment
            background_signal[sample_index:window_length+sample_index, channel_index] \
                = background_signal[sample_index:window_length+sample_index, channel_index] \
                + np.real(np.fft.ifft(background_ft, axis=0))

    # Truncate the signal to the original length
    background_signal = background_signal[0:number_samples, :]

    # Un-window the signal (just in case)
    background_signal = background_signal/sum(window_function[0:window_length:step_length])

    return background_signal


# Private functions
def _stft(audio_signal, window_function, step_length):
    """Short-time Fourier transform (STFT) (with zero-padding at the edges)"""

    # Number of samples and window length
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil((window_length-step_length+number_samples)/step_length))

    # Zero-padding at the start and end to center the windows
    audio_signal = np.pad(audio_signal, (window_length-step_length, number_times*step_length-number_samples),
                          'constant', constant_values=0)

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Window the signal
        sample_index = step_length*time_index
        audio_stft[:, time_index] = audio_signal[sample_index:window_length+sample_index]*window_function

    # Fourier transform of the frames
    audio_stft = np.fft.fft(audio_stft, axis=0)

    return audio_stft


def _istft(audio_stft, window_function, step_length):
    """Inverse short-time Fourier transform (STFT)"""

    # Window length and number of time frames
    window_length, number_times = np.shape(audio_stft)

    # Number of samples for the signal
    number_samples = (number_times-1)*step_length+window_length

    # Initialize the signal
    audio_signal = np.zeros(number_samples)

    # Inverse Fourier transform of the frames and real part to ensure real values
    audio_stft = np.real(np.fft.ifft(audio_stft, axis=0))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Constant overlap-add (if proper window and step)
        sample_index = step_length*time_index
        audio_signal[sample_index:window_length+sample_index] \
            = audio_signal[sample_index:window_length+sample_index]+audio_stft[:, time_index]

    # Remove the zero-padding at the start and end
    audio_signal = audio_signal[window_length-step_length:number_samples-(window_length-step_length)]

    # Un-apply window (just in case)
    audio_signal = audio_signal/sum(window_function[0:window_length:step_length])

    return audio_signal


def _acorr(data_matrix):
    """Autocorrelation using the Wiener–Khinchin theorem"""

    # Number of points in each column
    number_points = data_matrix.shape[0]

    # Power Spectral Density (PSD): PSD(X) = np.multiply(fft(X), conj(fft(X))) (after zero-padding for proper
    # autocorrelation)
    data_matrix = np.power(np.abs(np.fft.fft(data_matrix, n=2*number_points, axis=0)), 2)

    # Wiener–Khinchin theorem: PSD(X) = np.fft.fft(repet._acorr(X))
    autocorrelation_matrix = np.real(np.fft.ifft(data_matrix, axis=0))

    # Discard the symmetric part
    autocorrelation_matrix = autocorrelation_matrix[0:number_points, :]

    # Unbiased autocorrelation (lag 0 to number_points-1)
    autocorrelation_matrix = (autocorrelation_matrix.T/np.arange(number_points, 0, -1)).T

    return autocorrelation_matrix


def _beatspectrum(audio_spectrogram):
    """Beat spectrum using the autocorrelation"""

    # Autocorrelation of the frequency channels
    beat_spectrum = _acorr(audio_spectrogram.T)

    # Mean over the frequency channels
    beat_spectrum = np.mean(beat_spectrum, axis=1)

    return beat_spectrum


def _beatspectrogram(audio_spectrogram, segment_length, segment_step):
    """Beat spectrogram using the the beat spectrum"""

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Zero-padding the audio spectrogram to center the segments
    audio_spectrogram = np.pad(audio_spectrogram,
                               ((0, 0), (int(np.ceil((segment_length-1)/2)), int(np.floor((segment_length-1)/2)))),
                               'constant', constant_values=0)

    # Initialize beat spectrogram
    beat_spectrogram = np.zeros((segment_length, number_times))

    # Loop over the time frames (including the last one)
    for time_index in range(0, number_times, segment_step):

        # Beat spectrum of the centered audio spectrogram segment
        beat_spectrogram[:, time_index] = _beatspectrum(audio_spectrogram[:, time_index:time_index+segment_length])

        # Copy values in-between
        beat_spectrogram[:, time_index:min(time_index+segment_step-1, number_times)] \
            = beat_spectrogram[:, time_index:time_index+1]

    return beat_spectrogram


def _selfsimilaritymatrix(data_matrix):
    """Self-similarity matrix using the cosine similarity"""

    # Divide each column by its Euclidean norm
    data_matrix = data_matrix/np.sqrt(sum(np.power(data_matrix, 2), 0))

    # Multiply each normalized columns with each other
    similarity_matrix = np.matmul(data_matrix.T, data_matrix)

    return similarity_matrix


def _similaritymatrix(data_matrix1, data_matrix2):
    """Similarity matrix using the cosine similarity"""

    # Divide each column by its Euclidean norm
    data_matrix1 = data_matrix1/np.sqrt(sum(np.power(data_matrix1, 2), 0))
    data_matrix2 = data_matrix2/np.sqrt(sum(np.power(data_matrix2, 2), 0))

    # Multiply each normalized columns with each other
    similarity_matrix = np.matmul(data_matrix1.T, data_matrix2)

    return similarity_matrix


def _periods(beat_spectra, period_range):
    """Repeating periods from the beat spectra (spectrum or spectrogram)"""

    # The repeating periods are the indices of the maxima in the beat spectra for the period range (they do not account
    # for lag 0 and should be shorter than a third of the length as at least three segments are needed for the median)
    if beat_spectra.ndim == 1:
        repeating_periods = np.argmax(
            beat_spectra[period_range[0]:min(period_range[1], int(np.floor(beat_spectra.shape[0]/3)))]) + 1
    else:
        repeating_periods = np.argmax(
            beat_spectra[period_range[0]:min(period_range[1], int(np.floor(beat_spectra.shape[0]/3))), :], axis=0) + 1

    # Re-adjust the index or indices
    repeating_periods = repeating_periods + period_range[0]

    return repeating_periods


def _localmaxima(data_vector, minimum_value, minimum_distance, number_values):
    """Local maxima, values and indices"""

    # Number of data points
    number_data = len(data_vector)

    # Initialize maximum indices
    maximum_indices = np.array([], dtype=int)

    # Loop over the data points
    for data_index in range(0, number_data):

        # The local maximum should be greater than the maximum value
        if data_vector[data_index] >= minimum_value:

            # The local maximum should be strictly greater than the neighboring data points within +- minimum distance
            if all(data_vector[data_index] > data_vector[max(data_index-minimum_distance, 0):data_index]) \
                    and all(data_vector[data_index]
                            > data_vector[data_index+1:min(data_index+minimum_distance+1, number_data)]):

                # Save the maximum index
                maximum_indices = np.append(maximum_indices, data_index)

    # Sort the maximum values in descending order
    maximum_values = data_vector[maximum_indices]
    sort_indices = np.argsort(maximum_values)[::-1]

    # Keep only the top maximum values and indices
    number_values = min(number_values, len(maximum_values))
    maximum_values = maximum_values[0:number_values]
    maximum_indices = maximum_indices[sort_indices[0:number_values]].astype(int)

    return maximum_values, maximum_indices


def _indices(similarity_matrix, similarity_threshold, similarity_distance, similarity_number):
    """Similarity indices from the similarity matrix"""

    # Number of time frames
    number_times = similarity_matrix.shape[0]

    # Initialize the similarity indices
    similarity_indices = [None]*number_times

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Indices of the local maxima
        _, maximum_indices = \
            _localmaxima(similarity_matrix[:, time_index], similarity_threshold, similarity_distance, similarity_number)

        # Similarity indices for the current time frame
        similarity_indices[time_index] = maximum_indices

    return similarity_indices


def _mask(audio_spectrogram, repeating_period):
    """Repeating mask for REPET"""

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Number of repeating segments, including the last partial one
    number_segments = int(np.ceil(number_times/repeating_period))

    # Pad the audio spectrogram to have an integer number of segments and reshape it to a tensor
    audio_spectrogram = np.pad(audio_spectrogram, ((0, 0), (0, number_segments*repeating_period-number_times)),
                               'constant', constant_values=np.inf)
    audio_spectrogram = np.reshape(audio_spectrogram,
                                   (number_frequencies, repeating_period, number_segments), order='F')

    # Derive the repeating segment by taking the median over the segments, ignoring the nan parts
    repeating_segment = np.concatenate((
        np.median(audio_spectrogram[:, 0:number_times-(number_segments-1)*repeating_period, :], 2),
        np.median(audio_spectrogram[:, number_times-(number_segments-1)*repeating_period:repeating_period,
                  0:number_segments-1], 2)), 1)

    # Derive the repeating spectrogram by making sure it has less energy than the audio spectrogram
    repeating_spectrogram = np.minimum(audio_spectrogram, repeating_segment[:, :, np.newaxis])

    # Derive the repeating mask by normalizing the repeating spectrogram by the audio spectrogram
    repeating_mask = (repeating_spectrogram+np.finfo(float).eps)/(audio_spectrogram+np.finfo(float).eps)

    # Reshape the repeating mask and truncate to the original number of time frames
    repeating_mask = np.reshape(repeating_mask, (number_frequencies, number_segments*repeating_period), order='F')
    repeating_mask = repeating_mask[:, 0:number_times]

    return repeating_mask


def _adaptivemask(audio_spectrogram, repeating_periods, filter_order):
    """Repeating mask for the adaptive REPET"""

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Indices of the frames for the median filter centered on 0 (e.g., 3 => [-1,0,1], 4 => [-1,0,1,2], etc.)
    frame_indices = np.arange(1, filter_order+1)-int(np.ceil(filter_order/2))

    # Initialize the repeating spectrogram
    repeating_spectrogram = np.zeros((number_frequencies, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Indices of the frames for the median filter
        time_indices = time_index+frame_indices*repeating_periods[time_index]

        # Discard out-of-range indices
        time_indices = time_indices[np.logical_and(time_indices >= 0, time_indices < number_times)]

        # Median filter on the current time frame
        repeating_spectrogram[:, time_index] = np.median(audio_spectrogram[:, time_indices], 1)

    # Make sure the energy in the repeating spectrogram is smaller than in the audio spectrogram, for every
    # time-frequency bin
    repeating_spectrogram = np.minimum(audio_spectrogram, repeating_spectrogram)

    # Derive the repeating mask by normalizing the repeating spectrogram by the audio spectrogram
    repeating_mask = (repeating_spectrogram+np.finfo(float).eps)/(audio_spectrogram+np.finfo(float).eps)

    return repeating_mask


def _simmask(audio_spectrogram, similarity_indices):
    """Repeating mask for the REPET-SIM"""

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Initialize the repeating spectrogram
    repeating_spectrogram = np.zeros((number_frequencies, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Indices of the frames for the median filter
        time_indices = similarity_indices[time_index]

        # Median filter on the current time frame
        repeating_spectrogram[:, time_index] = np.median(audio_spectrogram[:, time_indices], 1)

    # Make sure the energy in the repeating spectrogram is smaller than in the audio spectrogram, for every
    # time-frequency bin
    repeating_spectrogram = np.minimum(audio_spectrogram, repeating_spectrogram)

    # Derive the repeating mask by normalizing the repeating spectrogram by the audio spectrogram
    repeating_mask = (repeating_spectrogram+np.finfo(float).eps)/(audio_spectrogram+np.finfo(float).eps)

    return repeating_mask
