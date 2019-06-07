import os
import glob
import scipy.io.wavfile
import soundfile
from parselmouth import PraatError

import config
import syllable_extractor as extractor
import audio_comparator as ac

data_dir = os.path.join(config.resourcesdir, 'Audio/')

algorithms = [ ac.duet,
               ac.ft2d,
               ac.hpss,
               ac.melodia,
               ac.repet,
               ac.repetsim,
               ac.rpca
             ]

def eval():
    eval = []

    for song in os.listdir(data_dir):

        eval_song = { 'name': song,
                      'lyric': -1,
                      'audio': { 'source': -1, 'test': [] } }

        # Get file paths
        path = os.path.join(data_dir, song)
        mix = os.path.join(path, 'mixture.wav')
        src = os.path.join(path, 'vocals.wav')

        # Get track information
        split_i = song.find('-')
        artist = song[:split_i-1]
        track = song[split_i+2:]

        print('{} - {}'.format(artist, track))
        print('---------------------------------------------------')

        # Do background strip and save as wav
        for algorithm in algorithms:
            output = algorithm(mix)
            for i, out in enumerate(output):
                fout = os.path.join(path, '{}_test_{}.wav'.format(algorithm.__name__, i))
                scipy.io.wavfile.write(fout, out[0], out[1].T)

        # Do audio syllable analysis
        eval_song['audio']['source'] = extractor.run(src)
        print('audio source:    {}'.format(eval_song['audio']['source']))

        for test in glob.glob(path + '/*test*.wav'):
            file = os.path.split(test)[1]

            try:
                # Check bit depth compatibility
                syl = extractor.run(test)
            except PraatError:
                # Use soundfile to convert file to PRAAT compatible bit depth
                signal, sample_rate = soundfile.read(test)
                soundfile.write(test, signal, sample_rate, subtype='PCM_16')
                syl = extractor.run(test)

            eval_song['audio']['test'].append(syl)
            print('{}: {}'.format(file, eval_song['audio']['test'][-1]))

        # Get lyrics
        import swaglyrics.cli
        lyrics = swaglyrics.cli.get_lyrics(track, artist, make_issue=False)
        print(lyrics)

song = 'Marvin Gaye - I Heard It Through the Grapevine'

eval_song = { 'name': song,
              'lyric': -1,
              'audio': { 'source': -1, 'test': [] } }

# Get file paths
path = os.path.join(data_dir, song)
mix = os.path.join(path, 'mixture.wav')
src = os.path.join(path, 'vocals.wav')

# Get track information
split_i = song.find('-')
artist = song[:split_i-1]
track = song[split_i+2:]

print('{} - {}'.format(artist, track))
print('---------------------------------------------------')

# Do background strip and save as wav

output = ac.msstorch(mix, apply_sparsity=False)
for i, out in enumerate(output):
    fout = os.path.join(path, '{}_test_{}.wav'.format(algorithm.__name__, i))
    scipy.io.wavfile.write(fout, out[0], out[1].T)

# Do audio syllable analysis
eval_song['audio']['source'] = extractor.run(src)
print('audio source:    {}'.format(eval_song['audio']['source']))

for test in glob.glob(path + '/*test*.wav'):
    file = os.path.split(test)[1]

    try:
        # Check bit depth compatibility
        syl = extractor.run(test)
    except PraatError:
        # Use soundfile to convert file to PRAAT compatible bit depth
        signal, sample_rate = soundfile.read(test)
        soundfile.write(test, signal, sample_rate, subtype='PCM_16')
        syl = extractor.run(test)

    eval_song['audio']['test'].append(syl)
    print('{}: {}'.format(file, eval_song['audio']['test'][-1]))

# Get lyrics
import swaglyrics.cli
lyrics = swaglyrics.cli.get_lyrics(track, artist, make_issue=False)
print(lyrics)
