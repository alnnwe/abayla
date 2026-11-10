import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SAQBOL_HASH_SECRET", "test-secret")


@pytest.fixture(scope="session")
def demo_csv(tmp_path_factory) -> Path:
    from saqbol.data.demo_dataset import write_demo_csv

    return write_demo_csv(tmp_path_factory.mktemp("data") / "demo.csv")


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, demo_csv) -> Path:
    """Обучает tfidf_wordchar на демо-данных во временном каталоге (не трогает artifacts/)."""
    from saqbol.training.train import main

    out = tmp_path_factory.mktemp("models")
    splits = tmp_path_factory.mktemp("splits")
    main(["--data", str(demo_csv), "--splits", str(splits), "--out", str(out), "--models", "tfidf_wordchar,rules"])
    return out


@pytest.fixture(scope="session")
def analyzer(trained_dir):
    from saqbol.service.analyzer import Analyzer

    return Analyzer.from_dir(trained_dir / "tfidf_wordchar")
