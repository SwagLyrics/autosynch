import os
import logging
import time
import msaf
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev
from swaglyrics.cli import get_lyrics

from autosynch.snd import SND
from autosynch.syllable_counter import SyllableCounter
from autosynch.mad_twinnet.scripts import twinnet
from autosynch.config import resources_dir

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

# Test info
songs = [ { 'song': 'Head Full of Doubt, Road Full of Promise',
       'artist': 'The Avett Brothers',
       'track_id': '7Kho44itYaCQZvZQVV2SLW', # can get this from SwSpotify later
       'genre': 'folk' }#,
        #   { 'song': 'Summertime',
        # 'artist': 'Ella Fitzgerald',
        # 'track_id': '6uqcvCdp6giqmdIppFkHk7',
        # 'genre': 'jazz' },
        #   { 'song': 'We Find Love',
        # 'artist': 'Daniel Caesar',
        # 'track_id': '1TPLsNVlofwX1txcE9gZZF',
        # 'genre': 'R&B' },
        #   { 'song': 'Man of the Year',
        # 'artist': 'ScHoolboy Q',
        # 'track_id': '5SsR3wtCOafDmZgvIdRhSm',
        # 'genre': 'rap' },
        #   { 'song': 'Bad Day',
        # 'artist': 'Daniel Powter',
        # 'track_id': '2Pwm2YtneLSWi7vyUpT5fs',
        # 'genre': 'pop' },
        #   { 'song': 'God is a woman',
        # 'artist': 'Ariana Grande',
        # 'track_id': '5OCJzvD7sykQEKHH7qAC3C',
        # 'genre': 'pop' },
        #   { 'song': 'See You Again',
        # 'artist': 'Tyler, the Creator',
        # 'track_id': '7KA4W4McWYRpgf0fWsJZWB',
        # 'genre': 'R&B' },
        #   { 'song': 'Your Man',
        # 'artist': 'Josh Turner',
        # 'track_id': '1WzAeadSKJhqykZFbJNmQv',
        # 'genre': 'country' },
        #   { 'song': "When I'm Gone",
        # 'artist': 'Eminem',
        # 'track_id': '42YNobZ4HN3tRVEA47wLT6',
        # 'genre': 'rap' }
        ]

# Settings
do_twinnet = True
boundary_algorithm = 'olda'
label_algorithm = 'fmc2d'
err_matrix = {'chorus': {'chorus': 0, 'verse': 5, 'other': 7},
              'verse':  {'chorus': 5, 'verse': 2, 'other': 5},
              'intro':  {'chorus': 3, 'verse': 5, 'other': 3},
              'bridge': {'chorus': 5, 'verse': 3, 'other': 3}}

# Module initializations
snd = SND(silencedb=-15)
sc = SyllableCounter()

# Perform MaD TwinNet in one batch
if do_twinnet:
    paths = [os.path.join(resources_dir, 'examples/' + song['track_id'] + '.wav') for song in songs]
    twinnet.twinnet_process(paths)

for song in songs:

    start_time = time.time()

    # Get file names
    mixed_path = os.path.join(resources_dir, 'examples/' + song['track_id'] + '.wav')
    voice_path = os.path.join(resources_dir, 'examples/' + song['track_id'] + '_voice.wav')

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
        if len(labels_density[label]) < 2:
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

    # Calculate accumulated error matrix
    dp = [[-1 for j in range(len(relabels))] for i in range(len(sc_syllables))]
    for i in range(len(sc_syllables)):
        for j in range(len(relabels)):
            dp[i][j] = err_matrix[sc_syllables[i][0]][relabels[j]]
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

    # Print
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

    print('Song: {}'.format(song['song']))
    print('Artist: {}'.format(song['artist']))
    print('Genre: {}'.format(song['genre']))
    print('Segmentation/labeling: {}/{}'.format(boundary_algorithm, label_algorithm))
    print('Genius syllables:')
    for i, n in enumerate(sc_syllables):
        print(i, n)
    print()
    print('Section boundaries:')
    for i, times in enumerate(zip(sections[:-1], sections[1:])):
        print(i, times[0], times[1])
    print()
    print('Alignment:')
    for section in alignment:
        print(section)
    print()
    print('Time taken: {}'.format(end_time-start_time))
