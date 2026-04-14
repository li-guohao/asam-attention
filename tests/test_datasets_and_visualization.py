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
