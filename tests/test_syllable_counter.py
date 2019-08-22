import pytest
from mock import mock, patch
from autosynch.syllable_counter import SyllableCounter

@pytest.fixture()
def counter():
    return SyllableCounter()

@patch.object(SyllableCounter, '_load_data', return_value=([],{}))
def test_load_data_on_init(mock_sc):
    SyllableCounter()
    assert mock_sc.called

def test_handles_no_lexicon_data():
    sc = SyllableCounter(sba_lexicon_path=None)
    assert sc.lexicon is None
    assert sc.counter is None

def test_handles_no_cmudict_path():
    sc = SyllableCounter(cmudict_path=None)

def test_naive(counter):
    assert counter._naive('ulotrichous') == 4
    assert counter._naive('borborygmus') == 4
    assert counter._naive('irritates') == 3
    assert counter._naive('qulliq') == 2

def test_sba(counter):
    assert counter._sba('quagmire') == 2
    assert counter._sba('subpoena') == 3
    assert counter._sba('cooperate') == 4
    assert counter._sba('footstool') == 2
    assert counter._sba('borscht') == 1
    assert counter._sba('qulliq') is None

def test_build_lyrics(counter):
    assert counter.build_lyrics('hello\nen-dash') == [('default', [['hello'], ['en', 'dash']])]
    assert counter.build_lyrics('[Chorus]\nhello\nhy-phen') == [('chorus', [['hello'], ['hy', 'phen']])]
    assert counter.build_lyrics('[Produced by X]\n[Chorus]\nhello\nsla/sh') == [('chorus', [['hello'], ['sla', 'sh']])]
    assert counter.build_lyrics('[Chorus]\nhello\n[Verse]\nits me') == [('chorus', [['hello']]), ('verse', [['its', 'me']])]
    assert counter.build_lyrics('[Bridge]\nhello\n[Intro]\nits me') == [('bridge', [['hello']]), ('intro', [['its', 'me']])]

def test_get_syllable_count_word(counter):
    assert counter.get_syllable_count_word('42') == 3
    assert counter.get_syllable_count_word('3.14') == 4
    assert counter.get_syllable_count_word('HELLO') == 2
    assert counter.get_syllable_count_word('don\'t~!@#$%^&*()-=_+`|}{":<>?.,/;"}') == 1
    assert counter.get_syllable_count_word('samantha\'s') == 3
    assert counter.get_syllable_count_word('qulliq') == 2

def test_get_syllable_count_lyrics(counter):
    assert counter.get_syllable_count_lyrics([('chorus', [['hElLO'], ['it\'s', 'M####e']])]) == [('chorus', [[2], [1, 1]])]

def test_get_syllable_count_per_section(counter):
    assert counter.get_syllable_count_per_section([('chorus', [[2], [1, 1]])]) == [('chorus', 4)]
