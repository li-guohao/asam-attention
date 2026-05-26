import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import torch

from asam import ASAMConfig, ASAMLayer

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


text_dataset = load_module("text_dataset_module", "datasets/text_dataset.py")
visualize_attention = load_module(
    "visualize_attention_module", "benchmarks/visualize_attention.py"
)

ArXivDataset = text_dataset.ArXivDataset
AGNewsDataset = text_dataset.AGNewsDataset
DBPediaDataset = text_dataset.DBPediaDataset
build_split_classification_tasks = text_dataset.build_split_classification_tasks
get_continual_dataloaders = text_dataset.get_continual_dataloaders
AttentionVisualizer = visualize_attention.AttentionVisualizer


def test_arxiv_build_text_and_metadata_label():
    item = {
        "title": "Learning Language Representations",
        "abstract": "A language model for translation and question answering.",
        "article": "This article studies natural language reasoning and text generation. "
        * 20,
        "categories": "cs.CL",
    }

    text = ArXivDataset._build_text(item)

    assert "Title: Learning Language Representations" in text
    assert "Abstract: A language model for translation and question answering." in text
    assert "Article:" in text
    assert ArXivDataset.CATEGORIES[ArXivDataset._infer_label(text, item)] == "cs.CL"


def test_arxiv_keyword_label_fallback_is_deterministic():
    text = (
        "finite element iterative solver numerical matrix partial differential equation "
        * 8
    )

    label_a = ArXivDataset._infer_label(text)
    label_b = ArXivDataset._infer_label(text)

    assert label_a == label_b
    assert ArXivDataset.CATEGORIES[label_a] == "math.NA"


def test_visualizer_extracts_real_attention_map():
    config = ASAMConfig(
        dim=64,
        num_heads=4,
        dim_head=16,
        pattern_type="hierarchical",
        use_adaptive_gate=True,
    )
    visualizer = AttentionVisualizer(ASAMLayer(config))
    x = torch.randn(1, 32, 64)

    attention_map, info = visualizer._extract_attention_map(x)

    assert attention_map.shape == (32, 32)
    assert np.isfinite(attention_map).all()
    assert np.allclose(attention_map.sum(axis=-1), 1.0, atol=1e-4)
    assert "gate_values" in info


def test_visualize_sparse_pattern_builds_pattern_without_forward(tmp_path):
    config = ASAMConfig(
        dim=64,
        num_heads=4,
        dim_head=16,
        pattern_type="hierarchical",
        use_adaptive_gate=False,
    )
    visualizer = AttentionVisualizer(ASAMLayer(config))
    output_path = tmp_path / "pattern.png"

    fig = visualizer.visualize_sparse_pattern(seq_len=32, save_path=str(output_path))

    assert fig is not None
    assert output_path.exists()


def test_agnews_dummy_dataset_and_split_tasks():
    dataset = AGNewsDataset._dummy_dataset(max_length=128, max_samples=16)
    tasks = build_split_classification_tasks(dataset, classes_per_task=2)

    assert len(tasks) == 2

    task0_tokens, task0_label, task0_id = tasks[0][0]
    task1_tokens, task1_label, task1_id = tasks[1][0]

    assert task0_tokens.shape[0] == 128
    assert task1_tokens.shape[0] == 128
    assert int(task0_label.item()) in {0, 1}
    assert int(task1_label.item()) in {0, 1}
    assert int(task0_id.item()) == 0
    assert int(task1_id.item()) == 1


def test_agnews_loader_supports_text_label_schema(monkeypatch):
    import sys
    import types

    fake_items = [
        {"text": "World news sample", "label": 0},
        {"text": "Sports news sample", "label": 1},
    ]

    fake_datasets = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: fake_items)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = AGNewsDataset.load(split="train", max_length=32, max_samples=2)

    assert len(dataset) == 2
    assert dataset.texts[0] == "World news sample"
    assert dataset.labels == [0, 1]



def test_split_dbpedia_continual_loader_builds_seven_tasks(monkeypatch):
    monkeypatch.setattr(DBPediaDataset, "load", classmethod(lambda cls, **kwargs: cls._dummy_dataset(max_length=64, max_samples=56)))

    train_loaders, val_loaders = get_continual_dataloaders(
        dataset_name="split_dbpedia",
        batch_size=4,
        max_length=64,
        classes_per_task=2,
        max_train_samples=56,
        max_val_samples=56,
    )

    assert len(train_loaders) == 7
    assert len(val_loaders) == 7

    batch_inputs, batch_labels, batch_task_ids = next(iter(train_loaders[-1]))
    assert batch_inputs.shape[-1] == 64
    assert batch_labels.max().item() <= 1
    assert int(batch_task_ids[0].item()) == 6



def test_split_arxiv_continual_loader_builds_four_tasks(monkeypatch):
    monkeypatch.setattr(ArXivDataset, "load", classmethod(lambda cls, **kwargs: cls._dummy_dataset(max_length=64, max_samples=32)))

    train_loaders, val_loaders = get_continual_dataloaders(
        dataset_name="split_arxiv",
        batch_size=4,
        max_length=64,
        classes_per_task=2,
        max_train_samples=32,
        max_val_samples=32,
    )

    assert len(train_loaders) == 4
    assert len(val_loaders) == 4

    batch_inputs, batch_labels, batch_task_ids = next(iter(train_loaders[0]))
    assert batch_inputs.shape[-1] == 64
    assert batch_labels.max().item() <= 1
    assert int(batch_task_ids[0].item()) == 0

