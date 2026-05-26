"""Tests for distributed training utilities."""
import torch
import pytest


def test_distributed_trainer_main_process():
    """DistributedTrainer identifies main process correctly outside torchrun."""
    from asam.distributed import DistributedTrainer

    trainer = DistributedTrainer()
    # When not launched via torchrun, is_main_process should return True
    assert trainer.is_main_process() is True


def test_distributed_trainer_no_init():
    """DistributedTrainer works without torch.distributed initialization."""
    from asam.distributed import DistributedTrainer

    trainer = DistributedTrainer()
    # Should not crash when torch.distributed is not initialized
    result = trainer.is_main_process()
    assert isinstance(result, bool)


def test_distributed_trainer_create_dataloader():
    """DistributedTrainer creates DataLoader without initialization."""
    from asam.distributed import DistributedTrainer
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
    trainer = DistributedTrainer()
    loader = trainer.create_dataloader(dataset, batch_size=4)
    assert len(loader) == 3  # ceil(10/4) = 3


def test_distributed_trainer_save_checkpoint():
    """DistributedTrainer saves checkpoint without error."""
    from asam.distributed import DistributedTrainer
    import torch.nn as nn
    import tempfile, os

    model = nn.Linear(10, 2)
    trainer = DistributedTrainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        trainer.save_checkpoint(model, path)
        assert os.path.exists(path)


def test_distributed_trainer_load_checkpoint():
    """DistributedTrainer loads checkpoint without error."""
    from asam.distributed import DistributedTrainer
    import torch.nn as nn
    import tempfile, os

    model = nn.Linear(10, 2)
    trainer = DistributedTrainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        trainer.save_checkpoint(model, path)
        epoch = trainer.load_checkpoint(model, path)
        assert isinstance(epoch, int)
