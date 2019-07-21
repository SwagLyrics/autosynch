import os
import logging
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from swaglyrics.cli import get_lyrics

from autosynch.snd import SND
from autosynch.syllable_counter import SyllableCounter
from autosynch.mad_twinnet.scripts import twinnet
from autosynch.config import resources_dir, spotify_oauth

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

# Test info
songs = [{ 'song': 'Head Full of Doubt, Road Full of Promise',
       'artist': 'The Avett Brothers',
       'track_id': '7Kho44itYaCQZvZQVV2SLW', # can get this from SwSpotify later
       'genre': 'folk' },
     { 'song': 'Summertime',
        'artist': 'Ella Fitzgerald',
        'track_id': '6uqcvCdp6giqmdIppFkHk7',
        'genre': 'jazz' },
     { 'song': 'Evergreen',
        'artist': 'YEBBA',
        'track_id': '4UuwnKoOKWYYnerZKZMtvD',
        'genre': 'singer-songerwriter' },
     { 'song': 'We Find Love',
        'artist': 'Daniel Caesar',
        'track_id': '1TPLsNVlofwX1txcE9gZZF',
        'genre': 'R&B' },
     { 'song': 'Man of the Year',
        'artist': 'ScHoolboy Q',
        'track_id': '5SsR3wtCOafDmZgvIdRhSm',
        'genre': 'rap' },
     { 'song': 'Bad Day',
        'artist': 'Daniel Powter',
        'track_id': '2Pwm2YtneLSWi7vyUpT5fs',
        'genre': 'pop' },
     { 'song': 'God is a woman',
        'artist': 'Ariana Grande',
        'track_id': '5OCJzvD7sykQEKHH7qAC3C',
        'genre': 'pop' },
     { 'song': 'See You Again',
        'artist': 'Tyler, the Creator',
        'track_id': '7KA4W4McWYRpgf0fWsJZWB',
        'genre': 'R&B' },
     { 'song': 'Your Man',
        'artist': 'Josh Turner',
        'track_id': '1WzAeadSKJhqykZFbJNmQv',
        'genre': 'country' },
     { 'song': "When I'm Gone",
        'artist': 'Eminem',
        'track_id': '42YNobZ4HN3tRVEA47wLT6',
        'genre': 'rap' }]

do_twinnet = False

# Module initializations
logging.info('Initializing modules...\n')
snd = SND(silencedb=-15)
sc = SyllableCounter()
client_credentials_manager = SpotifyClientCredentials(**spotify_oauth)
spotify = Spotify(client_credentials_manager=client_credentials_manager)

# Perform MaD TwinNet in one batch
if do_twinnet:
    logging.info('Beginning TwinNet...')
    paths = [os.path.join(resources_dir, 'align_tests/' + song['track_id'] + '.wav') for song in songs]
    twinnet.twinnet_process(paths)
    logging.info('TwinNet completed\n')

for song in songs:
    # Get file names
    out_path = os.path.join(resources_dir, 'align_tests/' + song['track_id'] + '_voice.wav')

    # Get lyrics from Genius
    lyrics = get_lyrics(song['song'], song['artist'])

    # Get syllable count from lyrics
    sc_syllables = sc.get_syllable_count_lyrics(lyrics)
    sc_per_section = [sum(sum(line) for line in section) for section in sc_syllables]

    # Get syllable count from SND
    snd_syllables = snd.run(out_path)

    # Get sections from Spotify
    track_sections = spotify.audio_analysis(song['track_id'])['sections']

    # Get SND syllables, density per section
    snd_per_section = []
    i_syl = 0
    for i, section in enumerate(track_sections):
        count = 0
        endpoint = section['start'] + section['duration']
        while i_syl < len(snd_syllables) and snd_syllables[i_syl] < endpoint:
            count += 1
            i_syl += 1
        snd_per_section.append((i, count, count/section['duration']))

    # Remove instrumental sections
