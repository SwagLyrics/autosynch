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

def line_align(songs, dump_dir, boundary_algorithm='olda',
               label_algorithm='fmc2d', do_twinnet=True):
    """
    Aligns given audio with lyrics by line. If dump_dir is None, no timestamp
    yml is created.

    :param songs: Song metadata in dict with keys 'song', 'artist', 'path' and \
                  'genre'. Key 'path' is audio file path. Key 'genre' optional.
    :type songs: list[dict{}] | dict{}
    :param dump_dir: Directory to store timestamp ymls.
    :type dump_dir: file-like | None
    :param boundary_algorithm: Segmentation algorithm for MSAF.
    :type boundary_algorithm: str
    :param label_algorithm: Labelling algorithm for MSAF.
    :type label_algorithm: str
    :param do_twinnet: Flag for performing vocal isolation.
    :type do_twinnet: bool
    :return align_data: List of alignment data. See below for formatting.
    :rtype: list[dict{}]
    """

    logging.info('Beginning alignment...')

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

    total_align_data = []

    for song in songs:

        logging.info('Processing {} by {}'.format(song['song'], song['artist']))

        start_time = time.time()

        # Get file names
        mixed_path = song['path']
        voice_path = os.path.splitext(song['path'])[0] + '_voice.wav'

        # Get lyrics from Genius
        lyrics = get_lyrics(song['song'], song['artist'])

        # Get syllable count from lyrics
        formatted_lyrics = sc.build_lyrics(lyrics)
        syl_lyrics = sc.get_syllable_count_lyrics(formatted_lyrics)
        sc_syllables = sc.get_syllable_count_per_section(syl_lyrics)

        # Get syllable count from SND
        snd_syllables = snd.run(voice_path)

        # Structural segmentation analysis on original audio
        sections, labels = msaf.process(mixed_path,
                                        boundaries_id=boundary_algorithm,
                                        labels_id=label_algorithm)

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

            duration = section[2] - section[1]
            density = count / duration

            # TODO: Improve instrumental categorization
            if density < 0.4:
                instrumentals.append(i)
            else:
                if section[0] not in labels_density:
                    labels_density[section[0]] = [[], []]
                labels_density[section[0]][0].append(count)
                labels_density[section[0]][1].append(density)
            # if section[0] not in labels_density:
            #     labels_density[section[0]] = [[], []]
            # labels_density[section[0]][0].append(count)
            # labels_density[section[0]][1].append(density)

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

            # TODO: Fix distance scales
            mean_syl = mean(labels_density[label][0])
            std_den  = stdev(labels_density[label][1])
            distance = sqrt(((mean_syl - gt_chorus_syl)/gt_chorus_syl)**2 + std_den**2)

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

        align_data = {'song': song['song'],
                      'artist': song['artist'],
                      'process time': end_time - start_time,
                      'duration': round((sections[-1] - sections[0]).item(), 2),
                      'align': []}

        if 'genre' in song:
            align_data['genre'] = song['genre']

        cur_lyric_section = -1
        for i, section in enumerate(alignment):
            for n, lyric_section in enumerate(section):
                if lyric_section != cur_lyric_section:
                    break_point = round((sections[i] + n * (sections[i+1]-sections[i])/len(section)).item(), 2)
                    if cur_lyric_section != 'instrumental' and align_data['align']:
                        align_data['align'][-1]['end'] = break_point
                    if lyric_section != 'instrumental':
                        align_data['align'].append({'label': sc_syllables[lyric_section][0],
                                                    'syllables': sc_syllables[lyric_section][1],
                                                    'start': break_point,
                                                    'lines': []})
                    cur_lyric_section = lyric_section

        if 'end' not in align_data['align'][-1]:
            align_data['align'][-1]['end'] = break_point

        for i, section in enumerate(align_data['align']):
            duration = section['end'] - section['start']
            line_start = section['start']
            for j, line in enumerate(formatted_lyrics[i][1]):
                line_text = ' '.join(line)
                line_syls = sum(syl_lyrics[i][1][j])
                line_duration = line_syls/align_data['align'][i]['syllables'] * duration

                align_data['align'][i]['lines'].append({'end': line_start + line_duration,
                                                        'text': line_text})

                line_start += line_duration

        if dump_dir is not None:
            file_name = '{}_{}.yml'.format(song['artist'], song['song']).replace(' ', '')
            file_path = os.path.join(dump_dir, file_name)

            with open(file_path, 'w') as f:
                yaml.dump(align_data, f, default_flow_style=False)

        total_align_data.append(align_data)

    return total_align_data

def eval_align(dump_dir, tagged_dir, out_file, verbose=False):
    """
    Evaluates segmentation based alignment by time error and percent coverage
    and outputs results to file. Must have tagged ymls in tagged_dir and have
    previously run line_align().

    :param dump_dir: Directory to store timestamp ymls.
    :type dump_dir: file-like
    :param tagged_dir: Directory with tagged timestamp ymls.
    :type tagged_dir: file-like
    :param out_file: Path for output file with results.
    :type out_file: file-like
    :param verbose: Flag for verbosity.
    :type verbose: bool
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
            with open(out_file, 'a') as f:
                f.write('{} - {}\n'.format(dump_data['artist'], dump_data['song']))
                f.write('Avg start error:       {}\n'.format(mean(song_err_start)))
                f.write('Avg start error (abs): {}\n'.format(mean(map(abs, song_err_start))))
                f.write('Avg end error:         {}\n'.format(mean(song_err_end)))
                f.write('Avg end error (abs):   {}\n'.format(mean(map(abs, song_err_end))))

                song_err_start.extend(song_err_end)
                f.write('Avg total error:       {}\n'.format(mean(song_err_start)))
                f.write('Std total error:       {}\n'.format(stdev(song_err_start)))

                song_err_start = list(map(abs, song_err_start))
                f.write('Avg total error (abs): {}\n'.format(mean(song_err_start)))
                f.write('Std total error (abs): {}\n'.format(stdev(song_err_start)))

                f.write('Percent coverage:      {}\n'.format(song_err_pcdur))
                f.write('\n')

    with open(out_file, 'a') as f:
        f.write('\n')
        f.write('Aggregate evaluation results\n')
        f.write('------------------------------------\n')
        f.write('Avg start error:       {}\n'.format(mean(total_err_start)))
        f.write('Avg start error (abs): {}\n'.format(mean(map(abs, total_err_start))))
        f.write('Avg end error:         {}\n'.format(mean(total_err_end)))
        f.write('Avg end error (abs):   {}\n'.format(mean(map(abs, total_err_end))))

        total_err_start.extend(total_err_end)
        f.write('Avg total error:       {}\n'.format(mean(total_err_start)))
        f.write('Std total error:       {}\n'.format(stdev(total_err_start)))

        total_err_start = list(map(abs, total_err_start))
        f.write('Avg total error (abs): {}\n'.format(mean(total_err_start)))
        f.write('Std total error (abs): {}\n'.format(stdev(total_err_start)))

        f.write('Avg percent coverage:  {}\n'.format(mean(total_err_pcdur)))
        f.write('Std percent coverage:  {}\n'.format(stdev(total_err_pcdur)))
        f.write('\n')

        f.write('Miscellaneous errors ({})\n'.format(len(misc_err)))
        f.write('------------------------------------\n')
        for error in misc_err:
            f.write(error + '\n')

        f.write('\n')

def iter_boundary_label_algorithms(songs, dump_dir, tagged_dir, evals_dir,
                                   do_twinnet=False, verbose=True):
    """
    Runs line_align() and eval_align() using each available segmentation and
    labelling algorithm in MSAF. See respective methods for parameter details.
    """

    for b_alg in msaf.get_all_boundary_algorithms():
        if b_alg == 'example':
            continue

        for l_alg in msaf.get_all_label_algorithms():
            out_base = '{}_{}.txt'.format(b_alg, l_alg)
            out_file = os.path.join(evals_dir, out_base)

            line_align(songs, dump_dir, b_alg, l_alg, do_twinnet)
            eval_align(dump_dir, tagged_dir, out_file, verbose)
