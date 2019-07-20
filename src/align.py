import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from swaglyrics.cli import get_lyrics

from snd import SND
from syllable_counter import SyllableCounter
from mad_twinnet.scripts import twinnet
from config import resources_dir, spotify_oauth


song = 'Head Full of Doubt, Road Full of Promise'
artist = 'The Avett Brothers'
track_id = '7Kho44itYaCQZvZQVV2SLW' # can get this from SwSpotify later
path = os.path.join(resources_dir, 'align_test.wav')


client_credentials_manager = SpotifyClientCredentials(**spotify_oauth)
spotify = Spotify(client_credentials_manager=client_credentials_manager)

snd = SND()
sc = SyllableCounter()

lyrics = get_lyrics(song, artist)
syllables = sc.get_syllable_count_lyrics(lyrics)

track_analysis = spotify.audio_analysis(track_id)
track_metadata = track_analysis['track']
track_sections = track_analysis['sections']

print(track_sections)
