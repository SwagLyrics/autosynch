import os
import sys
import argparse
import platform
import logging
import subprocess
import wave
import pyaudio
import yaml

from autosynch.align import line_align

def playback(audio_file, align_file, artist=None, song=None, save=None,
             chunk_size=1024, verbose=False):
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
    :param save: Dump directory for yml if align_file is None.
    :type save: file-like | None
    :param chunk_size: Buffer frames for playback stream.
    :type chunk_size: int
    :param verbose: Flag for printing logging info during alignment process.
    :type verbose: bool
    """

    # Perform alignment if necessary
    if align_file is None:
        if artist is None or song is None:
            raise ValueError('Params song and artist cannot be None if no align_file')
        if not verbose:
            logging.disable(logging.INFO)

        print('Processing...\n')
        print(audio_file)
        align = line_align({'song': song, 'artist': artist, 'path': audio_file}, save)[0]['align']
    else:
        with open(align_file, 'r') as f:
            align = yaml.safe_load(f)['align']

    # PyAudio setup
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

    # Stream audio and print lyrics
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

    print()

def mp3_to_wav(mp3_file):
    """
    Converts mp3 to wav using SoX. Requires SoX to be installed.

    :param mp3_file: Path to mp3 file.
    :type mp3_file: file-like
    :return wav_file: Path to wav file created.
    :rtype: str
    """

    wav_file = os.path.splitext(mp3_file)[0] + '.wav'
    subprocess.call(['sox', mp3_file, '-e', 'signed-integer', '-b', '16', wav_file])

    return wav_file

def main():
    parser = argparse.ArgumentParser(description='Play a song synchronized with its lyrics.')

    parser.add_argument('audio_file',
                        help='path to audio file to process')
    parser.add_argument('artist', nargs='?',
                        help='artist name: required if --align-file is not set')
    parser.add_argument('song', nargs='?',
                        help='song title: required if --align-file is not set')
    parser.add_argument('-f', '--align-file',
                        help='path to previously saved align file')
    parser.add_argument('-s', '--save', nargs='?', const=os.getcwd(),
                        metavar='SAVE_DIR', help='directory for saving align file')

    args = vars(parser.parse_args())

    if args['align_file'] is None and (args['artist'] is None or args['song'] is None):
        parser.error('artist and song are required if --align-file is not set')

    # Convert if mp3
    if os.path.splitext(args['audio_file'])[1] == '.mp3':
        args['audio_file'] = mp3_to_wav(args['audio_file'])

    playback(**args)

if __name__ == '__main__':
    main()
