"""Unit tests for the replay-memory and A-GEM helpers in the baseline comparison."""

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline = load_module("baseline_comparison_module", "experiments/run_baseline_comparison.py")


def test_sample_replay_memory_returns_batches():
    memory = baseline.SampleReplayMemory(max_samples=16)
    inputs = torch.arange(3 * 4).reshape(3, 4)
    labels = torch.tensor([0, 1, 0])
    memory.add_batch(inputs, labels)
    device = torch.device("cpu")
    sampled_inputs, sampled_labels = memory.sample(device, k=2)
    assert sampled_inputs.size(0) == 2
    assert sampled_labels.size(0) == 2
    assert memory.samples  # memory retained for later tasks


def test_project_gradients_makes_dot_nonnegative():
    # Construct main/ref gradients with a negative dot product.
    main = torch.tensor([1.0, 0.0], requires_grad=False)
    ref = torch.tensor([-1.0, 0.0], requires_grad=False)
    main_grads = [main.clone()]
    ref_grads = [ref.clone()]
    baseline.project_gradients(main_grads, ref_grads)
    projected = main_grads[0]
    assert torch.dot(projected, ref) >= 0.0
    # A non-negative dot product is left unchanged.
    main2 = torch.tensor([1.0, 0.0])
    ref2 = torch.tensor([1.0, 0.0])
    main_grads2 = [main2.clone()]
    ref_grads2 = [ref2.clone()]
    baseline.project_gradients(main_grads2, ref_grads2)
    assert torch.allclose(main_grads2[0], main2)
