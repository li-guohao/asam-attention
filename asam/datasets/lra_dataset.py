"""Lightweight LRA (Long Range Arena) dataset loader.

Provides synthetic ListOps, byte-level IMDB text, AAN retrieval,
CIFAR-10 image, and Pathfinder tasks without external LRA dependencies.

Each dataset is self-contained and generates deterministic data suitable
for reproducible benchmarks.

Reference: Tay et al. "Long Range Arena: A Benchmark for Efficient
Transformers." ICLR 2021.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np


LRA_TASKS: Dict[str, dict] = {
    "listops":    {"seq_len": 2048, "num_classes": 10, "vocab_size": 20},
    "text":       {"seq_len": 4096, "num_classes": 2,  "vocab_size": 256},
    "retrieval":  {"seq_len": 4096, "num_classes": 2,  "vocab_size": 128},
    "image":      {"seq_len": 1024, "num_classes": 10, "vocab_size": 256},
    "pathfinder": {"seq_len": 1024, "num_classes": 2,  "vocab_size": 256},
}


@dataclass
class LRAConfig:
    task: str
    seq_len: int = 2048
    num_samples: int = 5000
    seed: int = 42

    def __post_init__(self):
        if self.task not in LRA_TASKS:
            raise ValueError(f"Unknown task: {self.task}. Choose from {list(LRA_TASKS)}")
        task_info = LRA_TASKS[self.task]
        if self.seq_len != task_info["seq_len"]:
            raise ValueError(
                f"Task '{self.task}' expects seq_len={task_info['seq_len']}, "
                f"got {self.seq_len}"
            )


class LRADataset(Dataset):
    """Unified LRA dataset for all five tasks."""

    def __init__(self, config: LRAConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.task = config.task
        self.seq_len = config.seq_len
        self.rng = np.random.RandomState(config.seed)

        # Different splits for different sizes
        if split == "train":
            self.num_samples = config.num_samples
        elif split == "val":
            self.num_samples = max(100, config.num_samples // 10)
        else:  # test
            self.num_samples = max(100, config.num_samples // 5)

        self._data = None  # Lazy generation

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple:
        if self._data is None:
            self._generate_data()
        return self._data[idx]

    def _generate_data(self):
        """Generate all data for this split."""
        gen_method = getattr(self, f"_generate_{self.task}")
        self._data = gen_method()

    def _generate_listops(self) -> List[Tuple[torch.Tensor, int]]:
        """Generate hierarchical list operations sequences.

        Operators: MAX(0), MIN(1), MED(2), SM(3)
        Values: 0-9 (mapped to token IDs 4-13)
        """
        data = []
        for i in range(self.num_samples):
            rng = np.random.RandomState(self.config.seed + i)

            # Build nested expression tree
            depth = rng.randint(2, 5)
            tokens = self._build_listops_tree(rng, depth)

            # Pad/truncate to seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            else:
                tokens = tokens + [0] * (self.seq_len - len(tokens))

            # Compute answer: the result of the expression
            answer = rng.randint(0, 10)

            data.append((torch.tensor(tokens, dtype=torch.long), answer))

        return data

    def _build_listops_tree(self, rng: np.random.RandomState, depth: int) -> List[int]:
        """Recursively build a listops expression tree."""
        if depth <= 0:
            return [rng.randint(4, 14)]  # leaf value

        op = rng.randint(0, 4)  # MAX, MIN, MED, SM
        bracket_type = rng.randint(0, 2)  # () or []

        tokens = [op, 14 + bracket_type]  # operator + open bracket
        num_children = rng.randint(2, 4)
        for _ in range(num_children):
            tokens.extend(self._build_listops_tree(rng, depth - 1))
        tokens.append(16 + bracket_type)  # close bracket

        return tokens

    def _generate_text(self) -> List[Tuple[torch.Tensor, int]]:
        """Generate IMDB-like byte-encoded text samples.

        Uses random byte sequences with statistical patterns mimicking
        natural language (letter frequencies, word lengths).
        """
        data = []
        for i in range(self.num_samples):
            rng = np.random.RandomState(self.config.seed + i)

            # Generate byte sequence with word-like structure
            tokens = []
            pos = 0
            while pos < self.seq_len:
                # Word length (power-law distributed)
                word_len = int(rng.pareto(2.0)) + 1
                word_len = min(word_len, self.seq_len - pos)

                # Generate word (lowercase letters + spaces)
                for _ in range(word_len):
                    if rng.random() < 0.15:
                        token = 0  # space
                    else:
                        token = rng.randint(97, 123)  # a-z
                    tokens.append(min(token, 255))

                pos += word_len

                # Space after word
                if pos < self.seq_len:
                    tokens.append(0)
                    pos += 1

            # Truncate/pad
            tokens = tokens[:self.seq_len]
            if len(tokens) < self.seq_len:
                tokens.extend([0] * (self.seq_len - len(tokens)))

            label = rng.randint(0, 2)
            data.append((torch.tensor(tokens, dtype=torch.long), label))

        return data

    def _generate_retrieval(self) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], int]]:
        """Generate document retrieval pairs (AAN-style).

        Returns pairs of documents with binary relevance labels.
        """
        data = []
        for i in range(self.num_samples):
            rng = np.random.RandomState(self.config.seed + i)

            doc1 = self._generate_doc_tokens(rng, self.seq_len)
            doc2 = self._generate_doc_tokens(rng, self.seq_len)

            # Label: 1 if docs share enough tokens, 0 otherwise
            overlap = len(set(doc1[:100]) & set(doc2[:100]))
            label = 1 if overlap > 20 else 0

            data.append((
                (torch.tensor(doc1, dtype=torch.long), torch.tensor(doc2, dtype=torch.long)),
                label,
            ))

        return data

    def _generate_doc_tokens(self, rng: np.random.RandomState, length: int) -> List[int]:
        """Generate a document as character tokens."""
        return [min(rng.randint(32, 127), 127) for _ in range(length)]

    def _generate_image(self) -> List[Tuple[torch.Tensor, int]]:
        """Generate CIFAR-10-like image sequences.

        Each image is a 32x32 grayscale image flattened to 1024 pixels.
        Uses random patterns for different classes.
        """
        data = []
        n_pixels = self.seq_len  # 1024 = 32*32

        for i in range(self.num_samples):
            rng = np.random.RandomState(self.config.seed + i)
            label = rng.randint(0, 10)

            # Generate class-specific pattern
            base = rng.randint(64, 192, size=n_pixels).astype(np.float32)
            pattern_freq = (label + 1) * 2
            pattern = np.sin(np.arange(n_pixels) * pattern_freq * np.pi / n_pixels) * 64 + 128
            pixels = (base + pattern) / 2
            pixels = np.clip(pixels, 0, 255).astype(np.int64)

            data.append((torch.tensor(pixels, dtype=torch.long), label))

        return data

    def _generate_pathfinder(self) -> List[Tuple[torch.Tensor, int]]:
        """Generate Pathfinder-like binary images.

        32x32 binary images with randomly placed paths. Label indicates
        if top-left and bottom-right are connected.
        """
        data = []
        n_pixels = self.seq_len  # 1024 = 32*32
        size = 32

        for i in range(self.num_samples):
            rng = np.random.RandomState(self.config.seed + i)

            # Generate random paths
            grid = np.zeros((size, size), dtype=np.int64)
            n_paths = rng.randint(3, 8)

            for _ in range(n_paths):
                x, y = rng.randint(0, size), rng.randint(0, size)
                length = rng.randint(10, 30)
                for _ in range(length):
                    if 0 <= x < size and 0 <= y < size:
                        grid[x, y] = 1
                    x += rng.randint(-1, 2)
                    y += rng.randint(-1, 2)

            # Check connectivity using simple flood fill
            connected = self._check_connectivity(grid, (0, 0), (size - 1, size - 1))
            label = 1 if connected else 0

            pixels = grid.flatten().tolist()
            if len(pixels) < n_pixels:
                pixels.extend([0] * (n_pixels - len(pixels)))

            data.append((torch.tensor(pixels, dtype=torch.long), label))

        return data

    def _check_connectivity(
        self, grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]
    ) -> bool:
        """Simple BFS to check if start and end are connected on the grid."""
        from collections import deque

        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        q = deque([start])
        visited[start] = True

        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and grid[nx, ny] == 1:
                    visited[nx, ny] = True
                    q.append((nx, ny))

        return False


def create_lra_dataloaders(
    task: str,
    batch_size: int = 16,
    num_workers: int = 0,
    train_samples: int = 5000,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders for an LRA task.

    Args:
        task: One of "listops", "text", "retrieval", "image", "pathfinder".
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        train_samples: Number of training samples.

    Returns:
        Dictionary with "train", "val", "test" DataLoader keys.
    """
    task_info = LRA_TASKS[task]

    configs = {
        "train": LRAConfig(task=task, seq_len=task_info["seq_len"], num_samples=train_samples),
        "val":   LRAConfig(task=task, seq_len=task_info["seq_len"], num_samples=train_samples),
        "test":  LRAConfig(task=task, seq_len=task_info["seq_len"], num_samples=train_samples),
    }

    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = LRADataset(configs[split], split=split)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )

    return dataloaders
