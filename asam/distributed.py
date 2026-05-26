"""Distributed training utilities for ASAM models.

Supports DDP (DistributedDataParallel) wrapping with minimal boilerplate.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class DistributedTrainer:
    """Lightweight distributed training orchestrator.

    Handles torch.distributed lifecycle, process-safe logging,
    and checkpoint save/load for multi-GPU training.
    """

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self._initialized = False

    def is_main_process(self) -> bool:
        """Check if current process is rank 0."""
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def init_process_group(self):
        """Initialize distributed process group.

        Call this before any distributed operations.
        Use environment variables set by torchrun.
        """
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self.backend)
            self._initialized = True

    def cleanup(self):
        """Destroy the process group."""
        if self._initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self._initialized = False

    def wrap_ddp(
        self,
        model: nn.Module,
        find_unused_parameters: bool = False,
    ) -> DDP:
        """Wrap model with DistributedDataParallel."""
        if not torch.distributed.is_initialized():
            self.init_process_group()

        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        model = model.to(device)
        return DDP(
            model,
            device_ids=[torch.distributed.get_rank()],
            find_unused_parameters=find_unused_parameters,
        )

    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader with DistributedSampler if initialized."""
        sampler = None
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
    ):
        """Save checkpoint -- only on main process."""
        if not self.is_main_process():
            return

        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        model: nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """Load checkpoint, returning the epoch number."""
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint.get("epoch", 0)
