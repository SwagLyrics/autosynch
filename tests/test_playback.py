import os
import yaml
import pytest
from mock import mock, patch

from autosynch.config import tests_dir
from autosynch.playback import playback

@pytest.fixture(scope='session')
def audio_file():
    return os.path.join(tests_dir, 'resources/example_twinnet.wav')

@pytest.fixture(scope='session')
def align_file():
    return os.path.join(tests_dir, 'resources/example_align.yml')

@pytest.mark.timeout(70)
def test_playback_align_file(audio_file, align_file):
    playback(audio_file, align_file)

@patch('autosynch.playback.line_align')
def test_playback_no_align_file(mock_align, audio_file, align_file):
    with open(align_file, 'r') as f:
        mock_align.return_value = yaml.safe_load(f)

    playback(audio_file, None, artist='Liz Nelson', song='Rainfall')
    assert mock_align.called
