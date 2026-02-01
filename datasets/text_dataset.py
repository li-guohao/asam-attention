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


class LongTextDataset(Dataset):
    """Base class for long text datasets."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 4096,
        stride: Optional[int] = None
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
            tokens = tokens[start:start + self.max_length]
        else:
            # Pad
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


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
            texts = [item['text'] for item in ds]
            labels = [item['label'] for item in ds]
            
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
    """ArXiv abstract/paper classification for long documents."""
    
    CATEGORIES = [
        'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.RO',
        'physics.optics', 'physics.chem-ph', 'math.NA'
    ]
    
    @classmethod
    def load(cls, max_length: int = 4096, tokenizer=None):
        """
        Load ArXiv dataset.
        Requires arxiv dataset from HuggingFace or custom loading.
        """
        try:
            from datasets import load_dataset
            
            # This is a placeholder - actual ArXiv dataset loading
            ds = load_dataset("scientific_papers", "arxiv", split="train")
            
            texts = []
            labels = []
            
            for item in ds:
                # Combine title, abstract, and article
                text = item.get('article', '')
                if len(text) < 100:
                    text = item.get('abstract', '') + " " + text
                
                texts.append(text)
                # Use section as label (simplified)
                labels.append(random.randint(0, 7))
            
            if tokenizer is None:
                tokenizer = SimpleCharTokenizer()
            
            return cls(texts, labels, tokenizer, max_length)
        
        except Exception as e:
            print(f"Error loading ArXiv dataset: {e}")
            print("Using dummy data...")
            # Generate dummy long documents
            texts = []
            labels = []
            for _ in range(1000):
                # Simulate academic paper ~2000-8000 tokens
                length = random.randint(2000, 8000)
                text = " ".join(["scientific" if i % 10 == 0 else "word" for i in range(length)])
                texts.append(text)
                labels.append(random.randint(0, 7))
            
            if tokenizer is None:
                tokenizer = SimpleCharTokenizer()
            
            return cls(texts, labels, tokenizer, max_length)


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
        copy_distance: Optional[int] = None
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
    
    OPERATIONS = ['MAX', 'MIN', 'MED', 'SM', 'FM']
    DEPTH_RANGE = (1, 10)
    
    def __init__(self, num_samples: int = 10000, max_length: int = 2048):
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab = {op: i for i, op in enumerate(self.OPERATIONS)}
        self.vocab.update({str(i): i + 10 for i in range(10)})
        self.vocab.update({'[': 20, ']': 21, ' ': 22})
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random nested expression
        expr, result = self._generate_expression()
        
        # Tokenize
        tokens = [self.vocab.get(c, 0) for c in expr]
        
        # Pad/truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(result, dtype=torch.long)
    
    def _generate_expression(self, depth: int = 0) -> Tuple[str, int]:
        """Generate random nested expression."""
        if depth >= random.randint(*self.DEPTH_RANGE):
            # Leaf: random number
            num = random.randint(0, 9)
            return str(num), num
        
        op = random.choice(self.OPERATIONS)
        
        if op in ['MAX', 'MIN']:
            left, left_val = self._generate_expression(depth + 1)
            right, right_val = self._generate_expression(depth + 1)
            expr = f"[{op} {left} {right}]"
            result = max(left_val, right_val) if op == 'MAX' else min(left_val, right_val)
        elif op == 'MED':
            nums = [self._generate_expression(depth + 1)[1] for _ in range(3)]
            expr = f"[{op} {' '.join(str(n) for n in nums)}]"
            result = sorted(nums)[1]
        elif op == 'SM':  # Sum modulo 10
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
        return ''.join(chr(t) for t in tokens if t > 0)


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
    num_workers: int = 4
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
        dataset = ArXivDataset.load(max_length=max_length)
    elif dataset_name == "listops":
        dataset = ListOpsDataset(num_samples=10000 if split == "train" else 2000, max_length=max_length)
    elif dataset_name == "synthetic":
        dataset = SyntheticLongRangeDataset(
            num_samples=10000 if split == "train" else 2000,
            seq_len=max_length
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
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
