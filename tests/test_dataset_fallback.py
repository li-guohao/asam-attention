"""Tests for the gated synthetic-data fallback in datasets/text_dataset.py."""

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_ENV = "ASAM_ALLOW_DATASET_FALLBACK"


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


text_dataset = load_module("text_dataset_module", "datasets/text_dataset.py")


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def test_fallback_permission_raises_by_default(monkeypatch):
    monkeypatch.delenv(FALLBACK_ENV, raising=False)
    with pytest.raises(RuntimeError, match="ASAM_ALLOW_DATASET_FALLBACK"):
        text_dataset._require_fallback_permission("ag_news")


def test_fallback_permission_allowed_with_env(monkeypatch):
    monkeypatch.setenv(FALLBACK_ENV, "1")
    text_dataset._require_fallback_permission("ag_news")


def test_long_text_dataset_default_source():
    dataset = text_dataset.LongTextDataset(
        texts=["a"],
        labels=[0],
        tokenizer=text_dataset.SimpleCharTokenizer(),
        max_length=8,
    )
    assert dataset.data_source == "unknown"


def test_with_source_marks_fallback():
    dataset = text_dataset.AGNewsDataset._dummy_dataset(
        max_length=16,
        tokenizer=text_dataset.SimpleCharTokenizer(),
        max_samples=8,
    )
    tagged = text_dataset._with_source(dataset, "synthetic_fallback")
    assert tagged.data_source == "synthetic_fallback"


@pytest.mark.skipif(not _datasets_available(), reason="huggingface datasets not installed")
def test_agnews_real_load_marks_huggingface_source():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dataset = text_dataset.AGNewsDataset.load(
        split="train",
        max_length=16,
        tokenizer=text_dataset.SimpleCharTokenizer(),
        max_samples=8,
    )
    assert dataset.data_source == "huggingface"
