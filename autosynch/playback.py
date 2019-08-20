import sys
import wave
import pyaudio
import yaml
import logging

from autosynch.align import line_align

def playback(audio_file, align_file, artist=None, song=None, chunk_size=1024, verbose=False):
    """
    Plays audio with lyrics displayed at designated timestamp. If align_file is
    None, then full alignment process is performed with song and artist data.

    :param audio_file: Path to audio file.
    :type audio_file: file-like
    :param align_file: Path to timestamp yml, if existing.
    :type align_file: file-like | None
    :param artist: Artist name, if align_file does not exist.
    :type artist: str
    :param song: Song name, if align_file does not exist.
    :type song: str
    :param chunk_size: Buffer frames for playback stream.
    :type chunk_size: int
    """

    if align_file is None:
        if artist is None or song is None:
            raise ValueError('Params song and artist cannot be None if no align_file')
        if not verbose:
            logging.disable(logging.INFO)

        print('Processing...\n')
        align = line_align({'song': song, 'artist': artist, 'path': audio_file}, None)['align']
        print()

    else:
        with open(align_file, 'r') as f:
            align = yaml.safe_load(f)['align']

    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()
    sr = wf.getframerate()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=sr,
                    output=True)

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
    if len(sys.argv) == 4:
        playback(sys.argv[1], None, sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        playback(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python3 {} audio_file.wav artist_name song_title'.format(sys.argv[0]))
        sys.exit(1)
