"""Распознавание голосовых сообщений (отдельный необязательный модуль).

Пользователь САМ отправляет голосовой пересказ звонка; скрытое прослушивание звонков не реализуется.
Распознанный текст показывается пользователю и может быть исправлен до проверки.
Временный аудиофайл удаляется сразу после обработки (в блоке finally).
Задержка распознавания измеряется отдельно от задержки классификации.
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Transcript:
    text: str
    language: str | None
    asr_latency_s: float


class Transcriber(Protocol):
    def transcribe(self, path: str) -> Transcript: ...


class FasterWhisperTranscriber:
    """faster-whisper (CTranslate2-реализация Whisper). Устанавливается из requirements-voice.txt.

    Качество распознавания казахской речи и речи пожилых людей в этом проекте не измерялось —
    это нужно проверить в пилоте (см. docs/pilot_study_template.md)."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel  # импорт только при включённом голосе

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, path: str) -> Transcript:
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(path, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text=text, language=getattr(info, "language", None),
                          asr_latency_s=round(time.perf_counter() - t0, 3))


@contextmanager
def temporary_audio_path(suffix: str = ".ogg"):
    """Путь к временному файлу, который гарантированно удаляется после использования."""
    fd, path = tempfile.mkstemp(prefix="saqbol_voice_", suffix=suffix)
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


_transcriber: Transcriber | None = None


def get_transcriber(model_size: str) -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = FasterWhisperTranscriber(model_size)
    return _transcriber
