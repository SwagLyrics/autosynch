import sys
import wave
import pyaudio
import yaml
import logging

from autosynch.align import line_align

def playback(audio_file, align_file, chunk_size=1024):
    """
    Plays audio with lyrics displayed at designated timestamp.

    :param audio_file: Path to audio file.
    :type audio_file: file-like
    :param align_file: Path to timestamp yml, if existing.
    :type align_file: file-like
    :param chunk_size: Buffer frames for playback stream.
    :type chunk_size: int
    """

    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()
    sr = wf.getframerate()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=sr,
                    output=True)

    with open(align_file, 'r') as f:
        align = yaml.safe_load(f)['align']

    data = wf.readframes(chunk_size)

    n_frames = 0

    i_align = 0
    i_line = 0

    while data != b'':
        n_frames += chunk_size
        sec = n_frames / sr
        if i_align >= len(align) or sec < align[i_align]['start']:
            print('# instrumental', end='\033[K\r')
        else:
            print('# {}: {}'.format(align[i_align]['label'], align[i_align]['lines'][i_line]['text']), end='\033[K\r')
            if sec > align[i_align]['end']:
                i_align += 1
                i_line = 0
            elif sec > align[i_align]['lines'][i_line]['end']:
                i_line += 1

        stream.write(data)
        data = wf.readframes(chunk_size)

    stream.stop_stream()
    stream.close()

    p.terminate()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 {} audio_file.wav align_file.yml'.format(sys.argv[0]))
        sys.exit(1)

    audio_file = sys.argv[1]
    align_file = sys.argv[2]

    playback(audio_file, align_file)
