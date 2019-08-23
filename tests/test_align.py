import os
import pytest
from mock import mock, patch

from autosynch.config import tests_dir
from autosynch import align

@pytest.fixture(scope='session')
def song():
    return { 'song': 'Rainfall',
             'artist': 'Liz Nelson',
             'path': os.path.join(tests_dir, 'resources/example_twinnet.wav'),
             'genre': 'singer-songwriter'
           }

@patch('autosynch.align.get_lyrics')
def test_line_align(mock_lyrics, song, tmpdir):
    with open(os.path.join(tests_dir, 'resources/example_lyrics.txt'), 'r') as f:
        mock_lyrics.return_value = f.read()

    total_align_data = align.line_align(song, tmpdir, do_twinnet=False)
    assert len(total_align_data) > 0

    align_data = total_align_data[0]
    assert 'artist' in align_data
    assert 'song' in align_data
    assert 'genre' in align_data
    assert 'duration' in align_data
    assert 'process time' in align_data
    assert 'align' in align_data

    assert len(align_data['align']) > 0
    assert 'start' in align_data['align'][0]
    assert 'end' in align_data['align'][0]
    assert 'label' in align_data['align'][0]
    assert 'syllables' in align_data['align'][0]
    assert 'lines' in align_data['align'][0]

    assert len(align_data['align'][0]['lines']) > 0
    assert 'end' in align_data['align'][0]['lines'][0]
    assert 'text' in align_data['align'][0]['lines'][0]

    out_path = os.path.join(tmpdir, 'LizNelson_Rainfall.yml')
    assert os.path.isfile(out_path)

@patch('autosynch.mad_twinnet.scripts.twinnet.twinnet_process')
def test_line_align_with_twinnet(mock_twinnet, song):
    align.line_align([], None)
    assert mock_twinnet.called
