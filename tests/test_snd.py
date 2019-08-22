import os
import pytest
from mock import mock, patch

from autosynch.config import tests_dir
from autosynch.snd import SND

@pytest.fixture()
def snd():
    return SND(silencedb=-15)

def test_load_praat_script(snd):
    assert snd.script

def test_handles_bad_files(snd):
    assert snd.run('/non/existent/file') is None
    assert snd.run(os.path.realpath(__file__)) is None

def test_run_praat_script(snd):
    example_audio = os.path.join(tests_dir, 'resources/example_snd.wav')
    assert len(snd.run(example_audio)) == 10
