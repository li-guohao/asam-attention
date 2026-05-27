import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


text_dataset = load_module("text_dataset_provenance_module", "datasets/text_dataset.py")
AGNewsDataset = text_dataset.AGNewsDataset
DBPediaDataset = text_dataset.DBPediaDataset
ArXivDataset = text_dataset.ArXivDataset


def test_agnews_fallback_records_sanitized_provenance(monkeypatch, capsys):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets":
            raise ImportError("datasets missing; token hf_secret_value should not persist")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

    dataset = AGNewsDataset.load(split="train", max_length=32, max_samples=4)
    output = capsys.readouterr().out

    provenance = dataset.dataset_provenance
    assert provenance["source_kind"] == "fallback_synthetic"
    assert provenance["split"] == "train"
    assert provenance["sample_count"] == 4
    assert provenance["max_samples"] == 4
    assert "reason" in provenance
    serialized = json.dumps(provenance, sort_keys=True)
    assert "hf_secret_value" not in serialized
    assert "HF_TOKEN" not in serialized
    assert "hf_secret_value" not in output


def test_agnews_huggingface_success_records_non_secret_provenance(monkeypatch):
    fake_items = [
        {"title": "World title", "description": "World body", "label": 0},
        {"title": "Sports title", "description": "Sports body", "label": 1},
        {"title": "Business title", "description": "Business body", "label": 2},
    ]
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_items

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

    dataset = AGNewsDataset.load(split="test", max_length=32, max_samples=2)

    assert captured["args"] == ("ag_news",)
    assert captured["kwargs"]["split"] == "test"
    assert captured["kwargs"]["token"] == "hf_secret_value"
    provenance = dataset.dataset_provenance
    assert provenance["source_kind"] == "huggingface"
    assert provenance["dataset_name"] == "ag_news"
    assert provenance["dataset_config"] is None
    assert provenance["split"] == "test"
    assert provenance["sample_count"] == 2
    assert provenance["max_samples"] == 2
    assert "token" not in provenance
    assert "hf_secret_value" not in json.dumps(provenance, sort_keys=True)


def test_dbpedia_huggingface_success_records_dataset_provenance(monkeypatch):
    fake_items = [
        {"title": "Company title", "content": "Company body", "label": 0},
        {"title": "School title", "content": "School body", "label": 1},
        {"title": "Artist title", "content": "Artist body", "label": 2},
    ]
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_items

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

    dataset = DBPediaDataset.load(split="train", max_length=32, max_samples=2)

    assert captured["args"] == ("dbpedia_14",)
    assert captured["kwargs"]["split"] == "train"
    assert captured["kwargs"]["token"] == "hf_secret_value"
    provenance = dataset.dataset_provenance
    assert provenance["source_kind"] == "huggingface"
    assert provenance["dataset_name"] == "dbpedia_14"
    assert provenance["dataset_config"] is None
    assert provenance["split"] == "train"
    assert provenance["sample_count"] == 2
    assert provenance["max_samples"] == 2
    assert "hf_secret_value" not in json.dumps(provenance, sort_keys=True)


def test_arxiv_second_huggingface_candidate_records_actual_source(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "scientific_papers":
            raise RuntimeError("transient failure with hf_secret_value")
        return [
            {
                "title": "Fallback candidate paper",
                "abstract": "artificial intelligence reasoning " * 20,
                "article": "knowledge graph planning agent " * 80,
                "categories": "cs.AI",
            }
        ]

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

    dataset = ArXivDataset.load(
        split="validation",
        max_length=32,
        max_samples=1,
        min_text_length=64,
    )

    assert calls[0][0] == ("scientific_papers", "arxiv")
    assert calls[1][0] == ("ccdv/arxiv-summarization",)
    assert calls[1][1]["split"] == "validation"
    provenance = dataset.dataset_provenance
    assert provenance["source_kind"] == "huggingface"
    assert provenance["dataset_name"] == "ccdv/arxiv-summarization"
    assert provenance["dataset_config"] is None
    assert provenance["split"] == "validation"
    assert provenance["sample_count"] == 1
    assert provenance["max_samples"] == 1
    assert "hf_secret_value" not in json.dumps(provenance, sort_keys=True)
