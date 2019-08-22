import os
import pytest

from autosynch.config import tests_dir
from autosynch.mad_twinnet.scripts import twinnet

@pytest.fixture(scope='session')
def audio_file():
    return os.path.join(tests_dir, 'resources/example_twinnet.wav')

def test_handles_input_length_mismatch():
    assert twinnet.twinnet_process(['1', '2'], ['1']) is False

def test_generate_output_path(audio_file):
    assert twinnet._make_target_file_names(['1&*!!.wav']) == [['1&*!!_voice.wav', '1&*!!_bg_music.wav']]
    assert twinnet._make_target_file_names(['1', '2']) == [['1_voice.wav', '1_bg_music.wav'], ['2_voice.wav', '2_bg_music.wav']]

def test_twinnet_process_writes_to_file(audio_file, tmpdir):
    """Requires `mad.pt` in `mad_twinnet/outputs/states` to work"""

    out_base = os.path.splitext(os.path.basename(audio_file))[0] + '_voice.wav'
    out_path = os.path.join(str(tmpdir), out_base)

    assert twinnet.twinnet_process(audio_file, [[out_path, '']]) is True
    assert os.path.isfile(out_path)
