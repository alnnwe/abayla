import pandas as pd
import pytest

from saqbol.data.prepare import (augment_train, check_no_group_leakage, group_split, import_user_csv,
                                 load_dataset)
from saqbol.data.schema import COLUMNS, trainable, validate


def _row(**kw):
    base = dict(id="x1", text="Продиктуйте код", label="scam", scheme="bank_police_impersonation", language="ru",
                source_type="synthetic", source_id="t", template_group="g1", annotation_status="verified")
    base.update(kw)
    return base


def test_demo_dataset_is_valid(demo_csv):
    df = pd.read_csv(demo_csv, dtype=str)
    rep = validate(df)
    assert rep.ok, rep.errors
    assert set(df["source_type"]) == {"synthetic_demo"}
    assert {"ru", "kk"} <= set(df["language"])


@pytest.mark.parametrize("bad, fragment", [
    (dict(label="maybe"), "label"),
    (dict(scheme="unknown"), "scheme"),
    (dict(language="en"), "language"),
    (dict(text="   "), "пустой текст"),
    (dict(label="not_scam"), "not_scam со схемой"),
    (dict(scheme="none"), "scam без схемы"),
    (dict(annotation_status="ok"), "annotation_status"),
])
def test_validation_catches_errors(bad, fragment):
    df = pd.DataFrame([_row(**bad)])
    rep = validate(df)
    assert not rep.ok
    assert any(fragment in e for e in rep.errors), rep.errors


def test_validation_missing_columns():
    rep = validate(pd.DataFrame([{"text": "a"}]))
    assert not rep.ok and "нет обязательных колонок" in rep.errors[0]


def test_disputed_and_pending_excluded(demo_csv):
    df = load_dataset(demo_csv)
    tr = trainable(df)
    assert set(tr["annotation_status"]) == {"verified"}
    assert len(tr) < len(df)


def test_group_split_no_leakage_and_proportions(demo_csv):
    df = trainable(load_dataset(demo_csv))
    split = group_split(df, seed=42)
    check_no_group_leakage(split)
    groups = split.groupby("split")["template_group"].nunique()
    total = groups.sum()
    assert 0.6 < groups["train"] / total < 0.8
    for name in ("train", "val", "test"):
        part = split[split["split"] == name]
        assert set(part["label"]) == {"scam", "not_scam"}
    # перевод (kk) всегда в той же выборке, что и оригинал (ru)
    for _, grp in split.groupby("template_group"):
        assert grp["split"].nunique() == 1


def test_leakage_detector_fires():
    df = pd.DataFrame([_row(id="a", template_group="g"), _row(id="b", template_group="g")])
    df["split"] = ["train", "test"]
    with pytest.raises(AssertionError):
        check_no_group_leakage(df)


def test_augmentation_only_marks_train_rows_same_group(demo_csv):
    df = trainable(load_dataset(demo_csv)).head(10)
    aug = augment_train(df, seed=1)
    assert len(aug) == 20
    assert set(aug.loc[aug["augmented"] == 1, "template_group"]) <= set(df["template_group"])


def test_import_user_csv(tmp_path):
    p = tmp_path / "user.csv"
    pd.DataFrame({"message": ["Скиньте код", "Привет"], "is_fraud": ["1", "0"]}).to_csv(p, index=False)
    out = import_user_csv(p, text_col="message", label_col="is_fraud")
    assert list(out.columns) == COLUMNS
    assert out["label"].tolist() == ["scam", "not_scam"]
    assert out.loc[1, "scheme"] == "none"
    assert set(out["annotation_status"]) == {"pending"}  # импорт не считается проверенным
