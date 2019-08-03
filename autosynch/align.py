import os
import logging
import time
import msaf
import yaml
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev
from swaglyrics.cli import get_lyrics

from autosynch.snd import SND
from autosynch.syllable_counter import SyllableCounter
from autosynch.mad_twinnet.scripts import twinnet
from autosynch.config import resources_dir, dp_err_matrix

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

# Test info
songs = [ { 'song': 'Mine',
            'artist': 'Bazzi',
            'path': '/Users/Chris/autosynch/resources/align_tests/Bazzi_Mine.wav',
            'genre': 'pop' },
          { 'song': 'Finesse',
            'artist': 'Bruno Mars',
            'path': '/Users/Chris/autosynch/resources/align_tests/BrunoMars_Finesse.wav',
            'genre': 'funk' },
          { 'song': 'Please Me',
            'artist': 'Cardi B',
            'path': '/Users/Chris/autosynch/resources/align_tests/CardiB_PleaseMe.wav',
            'genre': 'hip hop' },
          { 'song': 'I Miss You',
            'artist': 'Clean Bandit',
            'path': '/Users/Chris/autosynch/resources/align_tests/CleanBandit_IMissYou.wav',
            'genre': 'electronic' },
          { 'song': 'Passionfruit',
            'artist': 'Drake',
            'path': '/Users/Chris/autosynch/resources/align_tests/Drake_Passionfruit.wav',
            'genre': 'hip hop' },
          { 'song': 'All the Stars',
            'artist': 'Kendrick Lamar',
            'path': '/Users/Chris/autosynch/resources/align_tests/KendrickLamar_AlltheStars.wav',
            'genre': 'rap' },
          { 'song': 'I Like Me Better',
            'artist': 'Lauv',
            'path': '/Users/Chris/autosynch/resources/align_tests/Lauv_ILikeMeBetter.wav',
            'genre': 'pop' },
          { 'song': 'Call Out My Name',
            'artist': 'The Weeknd',
            'path': '/Users/Chris/autosynch/resources/align_tests/TheWeeknd_CallOutMyName.wav',
            'genre': 'R&B' }
        ]

def seg_align(songs, dump_dir, boundary_algorithm='olda', label_algorithm='fmc2d', do_twinnet=True):
    """
    Performs segmentation-based alignment for songs listed on Genius.

    :param songs: Song info in dict format. Each song should be represented as \
                  a dict with keys 'song', 'artist', and 'path', representing \
                  the song name, artist, and audio file path respectively.
    :type songs: list[dict{}] | dict{}
    :param dump_dir: Path to directory to store alignment data.
    :type dump_dir: file-like
    :param boundary_algorithm: Segmentation algorithm name for MSAF.
    :type boundary_algorithm: str
    :param label_algorithm: Labelling algorithm name for MSAF.
    :type label_algorithm: str
    :param do_twinnet: Flag whether or not to perform vocal isolation.
    :type do_twinnet: bool
    """

    logging.info('Beginning segmentation alignment...')

    if isinstance(songs, dict):
        songs = [songs]

    # Module initializations
    snd = SND(silencedb=-15)
    sc = SyllableCounter()

    # Perform MaD TwinNet in one batch
    if do_twinnet:
        paths = [song['path'] for song in songs]
        twinnet.twinnet_process(paths)
    else:
        logging.info('Skipping MaD TwinNet')

    for song in songs:

        logging.info('Processing {} by {}'.format(song['song'], song['artist']))

        start_time = time.time()

        # Get file names
        mixed_path = song['path']
        voice_path = os.path.splitext(song['path'])[0] + '_voice.wav'

        # Get lyrics from Genius
        lyrics = get_lyrics(song['song'], song['artist'])

        # Get syllable count from lyrics
        sc_syllables = sc.get_syllable_count_per_section(lyrics)

        # Get syllable count from SND
        snd_syllables = snd.run(voice_path)

        # Structural segmentation analysis on original audio
        sections, labels = msaf.process(mixed_path, boundaries_id=boundary_algorithm, labels_id=label_algorithm)

        # Save instrumental section indices
        instrumentals = []

        # Get SND counts, densities per label
        max_count = 0

        labels_density = {}
        i_s = 0
        for i, section in enumerate(zip(labels, sections[:-1], sections[1:])):
            count = 0
            while i_s < len(snd_syllables) and snd_syllables[i_s] < section[2]:
                count += 1
                i_s += 1
            max_count = max(max_count, count)

            density = count/(section[2]-section[1])
            if density <= 0.7:
                instrumentals.append(i)
            else:
                if section[0] not in labels_density:
                    labels_density[section[0]] = [[], []]
                labels_density[section[0]][0].append(count)
                labels_density[section[0]][1].append(density)

        # Normalize SND syllable counts
        for label in labels_density:
            labels_density[label][0] = [count/max_count for count in labels_density[label][0]]

        # Normalize SSA syllable counts
        gt_max_syl = max(section[1] for section in sc_syllables)
        gt_chorus_syl = mean(section[1]/gt_max_syl for section in sc_syllables if section[0] == 'chorus')

        # Find label most similar to chorus
        min_label = labels[0]
        min_distance = float('inf')
        for label in labels_density:
            if len(labels_density[label][0]) < 2:
                continue

            mean_syl = mean(labels_density[label][0])
            std_den  = stdev(labels_density[label][1])
            distance = sqrt((mean_syl - gt_chorus_syl)**2 + std_den**2)

            if distance < min_distance:
                min_distance = distance
                min_label = label

        # Relabel
        relabels = [''] * len(labels)

        temp = defaultdict(list)
        for i, label in enumerate(labels):
            temp[label].append(i)
        for label in temp:
            for i in temp[label]:
                if i in instrumentals:
                    continue
                elif label == min_label:
                    relabels[i] = 'chorus'
                elif len(temp[label]) > 1:
                    relabels[i] = 'verse'
                else:
                    relabels[i] = 'other'
        del temp

        relabels = [label for label in relabels if label]
        if not relabels:
            logging.error('Whole song tagged as instrumental! Skipping...')
            continue

        # Calculate accumulated error matrix
        dp = [[-1 for j in range(len(relabels))] for i in range(len(sc_syllables))]
        for i in range(len(sc_syllables)):
            for j in range(len(relabels)):
                dp[i][j] = dp_err_matrix[sc_syllables[i][0]][relabels[j]]
                if i == 0 and j == 0:
                    pass
                elif i == 0:
                    dp[i][j] += dp[i][j-1]
                elif j == 0:
                    dp[i][j] += dp[i-1][j]
                else:
                    dp[i][j] += min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        # Backtrack
        i, j = len(sc_syllables)-1, len(relabels)-1
        path = []
        while True:
            path.append((i, j))
            if (i, j) == (0, 0):
                break
            elif i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_dir = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                if dp[i-1][j] == min_dir:
                    i -= 1
                elif dp[i][j-1] == min_dir:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
        path.reverse()

        # Process alignment and write to file
        alignment = [[] for i in range(len(labels))]
        for i in instrumentals:
            alignment[i].append('instrumental')

        section_id = 0
        j_prev = 0
        for (i, j) in path:
            if j != j_prev:
                section_id += 1
                j_prev = j
            while 'instrumental' in alignment[section_id]:
                section_id += 1
            alignment[section_id].append(i)

        end_time = time.time()

        print(labels)
        print(alignment)

        align_data = {'song': song['song'],
                      'artist': song['artist'],
                      'genre': song['genre'],
                      'process time': end_time - start_time,
                      'duration': round((sections[-1] - sections[0]).item(), 2),
                      'align': []}

        cur_lyric_section = -1
        for i, section in enumerate(alignment):
            for n, lyric_section in enumerate(section):
                if lyric_section != cur_lyric_section:
                    breakpoint = round((sections[i] + n * (sections[i+1]-sections[i])/len(section)).item(), 2)
                    if cur_lyric_section != 'instrumental' and align_data['align']:
                        align_data['align'][-1]['end'] = breakpoint
                    if lyric_section != 'instrumental':
                        align_data['align'].append({'label': sc_syllables[lyric_section][0],
                                                    'syllables': sc_syllables[lyric_section][1],
                                                    'start': breakpoint})
                    cur_lyric_section = lyric_section

        if 'end' not in align_data['align'][-1]:
            align_data['align'][-1]['end'] = breakpoint

        file_name = '{}_{}.yml'.format(song['artist'], song['song']).replace(' ', '')
        file_path = os.path.join(dump_dir, file_name)

        with open(file_path, 'w') as f:
            yaml.dump(align_data, f, default_flow_style=False)

def seg_align_eval(dump_dir, tagged_dir, verbose=False):
    """ Must have tagged yamls in tagged_dir and previously run seg_align().
    """
    
    total_err_start = []
    total_err_end = []
    total_err_pcdur = []
    misc_err = []

    for dump in os.listdir(dump_dir):
        dump_path = os.path.join(dump_dir, dump)
        tagged_path = os.path.join(tagged_dir, os.path.splitext(dump)[0] + '_tagged.yml')

        with open(dump_path, 'r') as d, open(tagged_path, 'r') as t:
            dump_data = yaml.safe_load(d)
            tagged_data = yaml.safe_load(t)

        song_err_start = []
        song_err_end = []
        song_err_pcdur = [0, 0]

        for i in range(len(tagged_data['align'])):
            dump_seg = dump_data['align'][i]
            tagged_seg = tagged_data['align'][i]

            if dump_seg['label'] != tagged_seg['label']:
                misc_err.append('{} - {}: Segmentation mismatch'.format(dump_data['artist'], dump_data['song']))
                continue

            song_err_start.append(dump_seg['start'] - tagged_seg['start'])
            song_err_end.append(dump_seg['end'] - tagged_seg['end'])

            song_err_pcdur[0] += max(0, min(dump_seg['end'], tagged_seg['end']) - max(dump_seg['start'], tagged_seg['start']))
            song_err_pcdur[1] += (tagged_seg['end'] - tagged_seg['start'])

        song_err_pcdur = song_err_pcdur[0]/song_err_pcdur[1]

        total_err_start.extend(song_err_start)
        total_err_end.extend(song_err_end)
        total_err_pcdur.append(song_err_pcdur)

        if verbose:
            print('{} - {}'.format(dump_data['artist'], dump_data['song']))
            print('Avg start error:       {}'.format(mean(song_err_start)))
            print('Avg start error (abs): {}'.format(mean(map(abs, song_err_start))))
            print('Avg end error:         {}'.format(mean(song_err_end)))
            print('Avg end error (abs):   {}'.format(mean(map(abs, song_err_end))))

            song_err_start.extend(song_err_end)
            print('Avg total error:       {}'.format(mean(song_err_start)))
            print('Std total error:       {}'.format(stdev(song_err_start)))

            song_err_start = list(map(abs, song_err_start))
            print('Avg total error (abs): {}'.format(mean(song_err_start)))
            print('Std total error (abs): {}'.format(stdev(song_err_start)))

            print('Percent coverage:      {}'.format(song_err_pcdur))
            print()

    print()
    print('Aggregate evaluation results')
    print('------------------------------------')
    print('Avg start error:       {}'.format(mean(total_err_start)))
    print('Avg start error (abs): {}'.format(mean(map(abs, total_err_start))))
    print('Avg end error:         {}'.format(mean(total_err_end)))
    print('Avg end error (abs):   {}'.format(mean(map(abs, total_err_end))))

    total_err_start.extend(total_err_end)
    print('Avg total error:       {}'.format(mean(total_err_start)))
    print('Std total error:       {}'.format(stdev(total_err_start)))

    total_err_start = list(map(abs, total_err_start))
    print('Avg total error (abs): {}'.format(mean(total_err_start)))
    print('Std total error (abs): {}'.format(stdev(total_err_start)))

    print('Avg percent coverage:  {}'.format(mean(total_err_pcdur)))
    print('Std percent coverage:  {}'.format(stdev(total_err_pcdur)))
    print()

    print('Miscellaneous errors ({})'.format(len(misc_err)))
    print('------------------------------------')
    for error in misc_err:
        print(error)

    print()

if __name__ == '__main__':
    seg_align(songs, '/Users/Chris/autosynch/resources/outputs', do_twinnet=False)
