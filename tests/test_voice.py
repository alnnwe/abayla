import os

import pytest

from abayla.voice.asr import temporary_audio_path


def test_temp_audio_removed_after_use():
    with temporary_audio_path() as p:
        open(p, "wb").write(b"fake-ogg")
        assert os.path.exists(p)
    assert not os.path.exists(p)


def test_temp_audio_removed_on_error():
    with pytest.raises(RuntimeError):
        with temporary_audio_path() as p:
            open(p, "wb").write(b"x")
            raise RuntimeError("asr crashed")
    assert not os.path.exists(p)
