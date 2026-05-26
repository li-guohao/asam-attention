import torch
import pytest


def test_listops_dataset():
    """ListOps returns correct shapes."""
    from asam.datasets.lra_dataset import LRADataset, LRAConfig

    config = LRAConfig(task="listops", seq_len=2048, num_samples=100)
    dataset = LRADataset(config, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    x, y = next(iter(loader))
    assert x.shape == (4, 2048)
    assert y.shape == (4,)
    assert x.dtype == torch.long
    assert y.min() >= 0 and y.max() < 10


def test_text_dataset():
    """IMDB text returns correct shapes."""
    from asam.datasets.lra_dataset import LRADataset, LRAConfig

    config = LRAConfig(task="text", seq_len=4096, num_samples=100)
    dataset = LRADataset(config, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    x, y = next(iter(loader))
    assert x.shape == (4, 4096)
    assert y.shape == (4,)
    assert y.min() >= 0 and y.max() < 2


def test_retrieval_dataset():
    """Retrieval returns correct shapes (two inputs)."""
    from asam.datasets.lra_dataset import LRADataset, LRAConfig

    config = LRAConfig(task="retrieval", seq_len=4096, num_samples=100)
    dataset = LRADataset(config, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    (x1, x2), y = next(iter(loader))
    assert x1.shape == (4, 4096)
    assert x2.shape == (4, 4096)
    assert y.shape == (4,)


def test_image_dataset():
    """CIFAR image returns correct shapes."""
    from asam.datasets.lra_dataset import LRADataset, LRAConfig

    config = LRAConfig(task="image", seq_len=1024, num_samples=100)
    dataset = LRADataset(config, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    x, y = next(iter(loader))
    assert x.shape == (4, 1024)
    assert y.shape == (4,)
    assert y.min() >= 0 and y.max() < 10


def test_pathfinder_dataset():
    """Pathfinder returns correct shapes."""
    from asam.datasets.lra_dataset import LRADataset, LRAConfig

    config = LRAConfig(task="pathfinder", seq_len=1024, num_samples=100)
    dataset = LRADataset(config, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    x, y = next(iter(loader))
    assert x.shape == (4, 1024)
    assert y.shape == (4,)
