"""
Text Classification Datasets for Long Sequence Testing
=======================================================

Supports:
- IMDB Reviews (standard)
- ArXiv Papers (long documents)
- BookCorpus (long narrative)
- PG-19 Books (very long sequences)
- Custom long-document datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
import json
import random
import hashlib
import re


class LongTextDataset(Dataset):
    """Base class for long text datasets."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 4096,
        stride: Optional[int] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate or pad
        if len(tokens) > self.max_length:
            # Random crop for training
            start = random.randint(0, len(tokens) - self.max_length)
            tokens = tokens[start : start + self.max_length]
        else:
            # Pad
            tokens = tokens + [0] * (self.max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


class IMDBLongDataset(LongTextDataset):
    """IMDB with extended sequence length support."""

    @classmethod
    def load(cls, split: str = "train", max_length: int = 2048, tokenizer=None):
        """
        Load IMDB dataset (requires torchtext or datasets library).

        Args:
            split: 'train' or 'test'
            max_length: Maximum sequence length
            tokenizer: Tokenizer function
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("imdb", split=split)
            texts = [item["text"] for item in ds]
            labels = [item["label"] for item in ds]

            if tokenizer is None:
                tokenizer = SimpleCharTokenizer()

            return cls(texts, labels, tokenizer, max_length)

        except ImportError:
            print("Please install datasets: pip install datasets")
            # Return dummy data for testing
            texts = ["This is a sample review. " * 100] * 100
            labels = [0, 1] * 50
            if tokenizer is None:
                tokenizer = SimpleCharTokenizer()
            return cls(texts, labels, tokenizer, max_length)


class ArXivDataset(LongTextDataset):
    """ArXiv-style long-document classification with deterministic labels."""

    CATEGORIES = [
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "cs.RO",
        "physics.optics",
        "physics.chem-ph",
        "math.NA",
    ]

    CATEGORY_KEYWORDS = {
        "cs.AI": (
            "artificial intelligence",
            "knowledge graph",
            "reasoning",
            "planning",
            "agent",
            "expert system",
        ),
        "cs.CL": (
            "natural language",
            "language model",
            "translation",
            "question answering",
            "text generation",
            "linguistic",
        ),
        "cs.CV": (
            "computer vision",
            "image",
            "object detection",
            "segmentation",
            "visual recognition",
            "video",
        ),
        "cs.LG": (
            "machine learning",
            "representation learning",
            "generalization",
            "optimization",
            "neural network",
            "training",
        ),
        "cs.RO": (
            "robot",
            "robotics",
            "manipulation",
            "navigation",
            "motion planning",
            "autonomous system",
        ),
        "physics.optics": (
            "optics",
            "photon",
            "laser",
            "spectroscopy",
            "interferometer",
            "waveguide",
        ),
        "physics.chem-ph": (
            "chemical physics",
            "quantum chemistry",
            "molecular dynamics",
            "reaction rate",
            "electronic structure",
            "molecule",
        ),
        "math.NA": (
            "numerical",
            "finite element",
            "iterative solver",
            "linear system",
            "partial differential equation",
            "matrix",
        ),
    }

    @classmethod
    def _clean_text(cls, value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = " ".join(str(item) for item in value if item)
        return re.sub(r"\s+", " ", str(value)).strip()

    @classmethod
    def _build_text(cls, item: Dict) -> str:
        item = dict(item)
        sections = []

        title = cls._clean_text(item.get("title"))
        if title:
            sections.append(f"Title: {title}")

        abstract = cls._clean_text(item.get("abstract") or item.get("summary"))
        if abstract:
            sections.append(f"Abstract: {abstract}")

        section_names = cls._clean_text(item.get("section_names"))
        if section_names:
            sections.append(f"Sections: {section_names}")

        article = cls._clean_text(
            item.get("article") or item.get("text") or item.get("paper")
        )
        if article:
            sections.append(f"Article: {article}")

        return "\n\n".join(sections).strip()

    @classmethod
    def _label_from_metadata(cls, item: Dict) -> Optional[int]:
        raw_categories = [
            item.get("categories"),
            item.get("category"),
            item.get("primary_category"),
            item.get("subject"),
            item.get("subjects"),
        ]

        for raw_value in raw_categories:
            if raw_value is None:
                continue

            if isinstance(raw_value, list):
                candidates = [str(value) for value in raw_value]
            else:
                candidates = re.split(r"[\s,;|]+", str(raw_value))

            for candidate in candidates:
                normalized = candidate.strip()
                if normalized in cls.CATEGORIES:
                    return cls.CATEGORIES.index(normalized)

        return None

    @classmethod
    def _infer_label(cls, text: str, item: Optional[Dict] = None) -> int:
        if item is not None:
            metadata_label = cls._label_from_metadata(item)
            if metadata_label is not None:
                return metadata_label

        lowered = text.lower()
        best_score = 0
        best_index = None

        for index, category in enumerate(cls.CATEGORIES):
            keywords = cls.CATEGORY_KEYWORDS.get(category, ())
            score = sum(lowered.count(keyword) for keyword in keywords)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score > 0:
            return best_index

        digest = hashlib.sha256(lowered.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % len(cls.CATEGORIES)

    @classmethod
    def _dummy_dataset(
        cls,
        max_length: int,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ):
        sample_count = max_samples or 256
        texts = []
        labels = []

        for sample_index in range(sample_count):
            label = sample_index % len(cls.CATEGORIES)
            category = cls.CATEGORIES[label]
            keywords = cls.CATEGORY_KEYWORDS[category]
            abstract = " ".join(keywords[:3])
            article_tokens = list(keywords) * 96
            article_body = " ".join(article_tokens)
            text = (
                f"Title: Synthetic {category} paper {sample_index}\n\n"
                f"Abstract: {abstract}. This fallback sample preserves long-context behavior.\n\n"
                f"Article: {article_body}"
            )
            texts.append(text)
            labels.append(label)

        if tokenizer is None:
            tokenizer = SimpleCharTokenizer()

        return cls(texts, labels, tokenizer, max_length)

    @classmethod
    def load(
        cls,
        split: str = "train",
        max_length: int = 4096,
        tokenizer=None,
        max_samples: Optional[int] = None,
        min_text_length: int = 256,
    ):
        """
        Load a real ArXiv-style long-document dataset.

        The loader first tries the original `scientific_papers/arxiv` dataset and
        then falls back to `ccdv/arxiv-summarization`, both from Hugging Face.
        Labels are inferred deterministically from metadata or paper content.
        """
        if tokenizer is None:
            tokenizer = SimpleCharTokenizer()

        try:
            from datasets import load_dataset
        except ImportError:
            print("datasets not installed, using fallback ArXiv samples...")
            return cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples)

        dataset_candidates = [
            ("scientific_papers", "arxiv"),
            ("ccdv/arxiv-summarization", None),
        ]
        load_errors = []

        for dataset_name, dataset_config in dataset_candidates:
            try:
                if dataset_config is None:
                    ds = load_dataset(dataset_name, split=split)
                else:
                    ds = load_dataset(dataset_name, dataset_config, split=split)

                texts = []
                labels = []

                for item in ds:
                    text = cls._build_text(item)
                    if len(text) < min_text_length:
                        continue

                    texts.append(text)
                    labels.append(cls._infer_label(text, item))

                    if max_samples is not None and len(texts) >= max_samples:
                        break

                if texts:
                    return cls(texts, labels, tokenizer, max_length)

                load_errors.append(f"{dataset_name}: no valid samples after filtering")
            except Exception as exc:
                load_errors.append(f"{dataset_name}: {exc}")

        print("Error loading ArXiv datasets; using fallback samples...")
        for error in load_errors:
            print(f"  - {error}")
        return cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples)


class SyntheticLongRangeDataset(Dataset):
    """
    Synthetic dataset for testing long-range dependencies.

    Task: Copy the first token to the end after a long delay.
    Tests the model's ability to maintain information over long sequences.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 4096,
        vocab_size: int = 100,
        copy_distance: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.copy_distance = copy_distance or seq_len - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create sequence with copy task
        sequence = torch.randint(1, self.vocab_size, (self.seq_len,))

        # Mark a special token to copy
        copy_token = torch.randint(1, self.vocab_size, (1,))
        sequence[0] = copy_token

        # Target is the copy token
        target = copy_token

        return sequence, target


class ListOpsDataset(Dataset):
    """
    ListOps dataset for hierarchical reasoning.

    Format: Nested list operations like [MAX [MIN 3 4] 5]
    """

    OPERATIONS = ["MAX", "MIN", "MED", "SM", "FM"]
    DEPTH_RANGE = (1, 10)

    def __init__(self, num_samples: int = 10000, max_length: int = 2048):
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab = {op: i for i, op in enumerate(self.OPERATIONS)}
        self.vocab.update({str(i): i + 10 for i in range(10)})
        self.vocab.update({"[": 20, "]": 21, " ": 22})

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random nested expression
        expr, result = self._generate_expression()

        # Tokenize
        tokens = [self.vocab.get(c, 0) for c in expr]

        # Pad/truncate
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            result, dtype=torch.long
        )

    def _generate_expression(self, depth: int = 0) -> Tuple[str, int]:
        """Generate random nested expression."""
        if depth >= random.randint(*self.DEPTH_RANGE):
            # Leaf: random number
            num = random.randint(0, 9)
            return str(num), num

        op = random.choice(self.OPERATIONS)

        if op in ["MAX", "MIN"]:
            left, left_val = self._generate_expression(depth + 1)
            right, right_val = self._generate_expression(depth + 1)
            expr = f"[{op} {left} {right}]"
            result = (
                max(left_val, right_val) if op == "MAX" else min(left_val, right_val)
            )
        elif op == "MED":
            nums = [self._generate_expression(depth + 1)[1] for _ in range(3)]
            expr = f"[{op} {' '.join(str(n) for n in nums)}]"
            result = sorted(nums)[1]
        elif op == "SM":  # Sum modulo 10
            nums = [self._generate_expression(depth + 1)[1] for _ in range(3)]
            expr = f"[{op} {' '.join(str(n) for n in nums)}]"
            result = sum(nums) % 10
        else:  # FM: First modulo second
            left, left_val = self._generate_expression(depth + 1)
            right, right_val = self._generate_expression(depth + 1)
            expr = f"[{op} {left} {right}]"
            result = left_val % (right_val + 1)

        return expr, result


class SimpleCharTokenizer:
    """Simple character-level tokenizer for testing."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens if t > 0)


class BPETokenizer:
    """BPE tokenizer wrapper (requires tokenizers library)."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers

            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            self.trainer = trainers.BpeTrainer(vocab_size=vocab_size)
        except ImportError:
            print("tokenizers not installed, using SimpleCharTokenizer")
            self.tokenizer = SimpleCharTokenizer(vocab_size)

    def encode(self, text: str) -> List[int]:
        if isinstance(self.tokenizer, SimpleCharTokenizer):
            return self.tokenizer.encode(text)
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: List[int]) -> str:
        if isinstance(self.tokenizer, SimpleCharTokenizer):
            return self.tokenizer.decode(tokens)
        return self.tokenizer.decode(tokens)

    def train(self, texts: List[str]):
        if not isinstance(self.tokenizer, SimpleCharTokenizer):
            self.tokenizer.train_from_iterator(texts, trainer=self.trainer)


def get_dataloader(
    dataset_name: str,
    batch_size: int = 8,
    max_length: int = 4096,
    split: str = "train",
    num_workers: int = 4,
) -> DataLoader:
    """
    Get dataloader for specified dataset.

    Args:
        dataset_name: One of ['imdb', 'arxiv', 'listops', 'synthetic']
        batch_size: Batch size
        max_length: Maximum sequence length
        split: Dataset split
        num_workers: Number of data loading workers
    """

    if dataset_name == "imdb":
        dataset = IMDBLongDataset.load(split=split, max_length=max_length)
    elif dataset_name == "arxiv":
        dataset = ArXivDataset.load(split=split, max_length=max_length)
    elif dataset_name == "listops":
        dataset = ListOpsDataset(
            num_samples=10000 if split == "train" else 2000, max_length=max_length
        )
    elif dataset_name == "synthetic":
        dataset = SyntheticLongRangeDataset(
            num_samples=10000 if split == "train" else 2000, seq_len=max_length
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test datasets
    print("Testing ListOps dataset...")
    ds = ListOpsDataset(num_samples=10, max_length=512)
    for i in range(3):
        tokens, label = ds[i]
        print(f"Sample {i}: tokens shape={tokens.shape}, label={label}")

    print("\nTesting Synthetic dataset...")
    ds = SyntheticLongRangeDataset(num_samples=10, seq_len=512)
    for i in range(3):
        seq, target = ds[i]
        print(f"Sample {i}: seq shape={seq.shape}, target={target}")
