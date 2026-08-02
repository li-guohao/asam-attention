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

ASAM_FALLBACK_ENV = "ASAM_ALLOW_DATASET_FALLBACK"


def _require_fallback_permission(dataset_name: str) -> None:
    """Raise unless the synthetic dataset fallback is explicitly allowed."""
    allowed = os.getenv(ASAM_FALLBACK_ENV, "0").strip().lower() in {"1", "true", "yes"}
    if not allowed:
        raise RuntimeError(
            f"Failed to load real '{dataset_name}' data; the synthetic fallback is disabled. "
            f"Set {ASAM_FALLBACK_ENV}=1 to explicitly allow keyword-synthesized dummy data, "
            "or fix the data source (install the 'datasets' package / network / cache)."
        )
    print(
        f"WARNING: using synthetic fallback data for '{dataset_name}' "
        f"(enabled via {ASAM_FALLBACK_ENV}=1)."
    )


def _with_source(dataset: "LongTextDataset", source: str) -> "LongTextDataset":
    dataset.data_source = source
    return dataset


class LongTextDataset(Dataset):
    """Base class for long text datasets."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 4096,
        stride: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2
        self.vocab_size = vocab_size or 10000
        self.data_source = "unknown"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer.encode(text)
        tokens = [min(t, self.vocab_size - 1) for t in tokens]

        if len(tokens) > self.max_length:
            max_start = len(tokens) - self.max_length
            crop_key = f"{idx}:{text}".encode("utf-8", errors="ignore")
            start = int(hashlib.sha256(crop_key).hexdigest(), 16) % (max_start + 1)
            tokens = tokens[start : start + self.max_length]
        else:
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

            return _with_source(cls(texts, labels, tokenizer, max_length), "huggingface")

        except Exception as exc:
            print(f"Error loading IMDB ({type(exc).__name__}: {exc}); fallback disabled by default.")
            _require_fallback_permission("imdb")
            texts = ["This is a sample review. " * 100] * 100
            labels = [0, 1] * 50
            if tokenizer is None:
                tokenizer = SimpleCharTokenizer()
            return _with_source(cls(texts, labels, tokenizer, max_length), "synthetic_fallback")


class AGNewsDataset(LongTextDataset):
    """AG News dataset with optional fallback samples."""

    LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
    LABEL_KEYWORDS = {
        0: ("world", "government", "country", "election", "diplomatic", "war"),
        1: ("sports", "team", "match", "coach", "league", "tournament"),
        2: ("business", "market", "stock", "company", "economy", "trade"),
        3: ("technology", "science", "software", "internet", "device", "research"),
    }

    @classmethod
    def _dummy_dataset(cls, max_length: int, tokenizer=None, max_samples: Optional[int] = None):
        sample_count = max_samples or 256
        texts = []
        labels = []

        for sample_index in range(sample_count):
            label = sample_index % len(cls.LABEL_NAMES)
            keywords = cls.LABEL_KEYWORDS[label]
            headline = f"{cls.LABEL_NAMES[label]} headline {sample_index}"
            body = " ".join(list(keywords) * 48)
            texts.append(f"Headline: {headline}\n\nArticle: {body}")
            labels.append(label)

        if tokenizer is None:
            tokenizer = SimpleCharTokenizer()

        return cls(texts, labels, tokenizer, max_length)

    @classmethod
    def load(
        cls,
        split: str = "train",
        max_length: int = 2048,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ):
        if tokenizer is None:
            tokenizer = SimpleCharTokenizer()

        try:
            from datasets import load_dataset
        except ImportError as exc:
            print(f"datasets not installed ({exc}); fallback disabled by default.")
            _require_fallback_permission("ag_news")
            return _with_source(
                cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples),
                "synthetic_fallback",
            )

        try:
            hf_token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
                or os.getenv("HF_HUB_TOKEN")
            )
            ds = load_dataset("ag_news", split=split, token=hf_token)
            if max_samples is None:
                texts = []
                labels = []
                for item in ds:
                    if "title" in item and "description" in item:
                        article_text = f"Headline: {item['title']}\n\nArticle: {item['description']}"
                    elif "text" in item:
                        article_text = str(item["text"])
                    else:
                        raise KeyError("AG News item must contain either title/description or text")
                    texts.append(article_text)
                    labels.append(int(item["label"]))
            else:
                num_labels = max(1, len(cls.LABEL_NAMES))
                target_per_label = max(1, (max_samples + num_labels - 1) // num_labels)
                label_buckets = {label: [] for label in range(num_labels)}
                for item in ds:
                    label = int(item["label"])
                    if label not in label_buckets or len(label_buckets[label]) >= target_per_label:
                        continue
                    if "title" in item and "description" in item:
                        article_text = f"Headline: {item['title']}\n\nArticle: {item['description']}"
                    elif "text" in item:
                        article_text = str(item["text"])
                    else:
                        raise KeyError("AG News item must contain either title/description or text")
                    label_buckets[label].append(article_text)
                    if all(len(bucket) >= target_per_label for bucket in label_buckets.values()):
                        break

                texts = []
                labels = []
                for label in range(num_labels):
                    for article_text in label_buckets[label]:
                        texts.append(article_text)
                        labels.append(label)
                        if len(texts) >= max_samples:
                            break
                    if len(texts) >= max_samples:
                        break
            return _with_source(cls(texts, labels, tokenizer, max_length), "huggingface")
        except Exception as exc:
            print(f"Error loading AG News ({type(exc).__name__}: {exc}); fallback disabled by default.")
            _require_fallback_permission("ag_news")
            return _with_source(
                cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples),
                "synthetic_fallback",
            )



class DBPediaDataset(LongTextDataset):
    """DBPedia ontology classification with balanced sampling and fallback samples."""

    LABEL_NAMES = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork",
    ]

    SHARED_KEYWORDS = (
        "entity",
        "encyclopedia",
        "reference",
        "history",
        "description",
        "classification",
    )

    LABEL_KEYWORDS = {
        0: ("company", "business", "industry", "founded", "market", "corporation"),
        1: ("school", "university", "college", "campus", "faculty", "education"),
        2: ("artist", "painting", "sculpture", "gallery", "exhibition", "creative"),
        3: ("athlete", "sport", "team", "season", "championship", "competition"),
        4: ("office", "government", "minister", "parliament", "election", "policy"),
        5: ("transport", "vehicle", "engine", "rail", "aircraft", "passenger"),
        6: ("building", "architecture", "tower", "structure", "construction", "landmark"),
        7: ("river", "mountain", "lake", "forest", "region", "geography"),
        8: ("village", "town", "municipality", "population", "district", "settlement"),
        9: ("animal", "species", "habitat", "mammal", "genus", "wildlife"),
        10: ("plant", "botanical", "flower", "leaf", "species", "cultivation"),
        11: ("album", "music", "track", "recording", "release", "song"),
        12: ("film", "cinema", "director", "screenplay", "actor", "release"),
        13: ("book", "novel", "author", "publication", "chapter", "literature"),
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
        title = cls._clean_text(item.get("title") or item.get("name"))
        content = cls._clean_text(
            item.get("content") or item.get("text") or item.get("article") or item.get("description")
        )
        sections = []
        if title:
            sections.append(f"Title: {title}")
        if content:
            sections.append(f"Content: {content}")
        return "\n\n".join(sections).strip()

    @classmethod
    def _dummy_dataset(
        cls,
        max_length: int,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ):
        sample_count = max_samples or 280
        texts = []
        labels = []

        for sample_index in range(sample_count):
            label = sample_index % len(cls.LABEL_NAMES)
            label_name = cls.LABEL_NAMES[label]
            keywords = cls.LABEL_KEYWORDS[label]
            shared_prefix = " ".join(list(cls.SHARED_KEYWORDS) * 8)
            category_body = " ".join(list(keywords) * 40)
            bridge_tokens = " ".join(list(keywords[:3]) * 12)
            text = (
                f"Title: Synthetic {label_name} entry {sample_index}\n\n"
                f"Content: {shared_prefix} {bridge_tokens}. "
                f"This fallback entry describes a {label_name.lower()} example with long-context evidence. "
                f"{category_body}"
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
        max_length: int = 2048,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ):
        if tokenizer is None:
            tokenizer = SimpleCharTokenizer()

        try:
            from datasets import load_dataset
        except ImportError as exc:
            print(f"datasets not installed ({exc}); fallback disabled by default.")
            _require_fallback_permission("dbpedia_14")
            return _with_source(
                cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples),
                "synthetic_fallback",
            )

        try:
            hf_token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
                or os.getenv("HF_HUB_TOKEN")
            )
            ds = load_dataset("dbpedia_14", split=split, token=hf_token)
            if max_samples is None:
                texts = []
                labels = []
                for item in ds:
                    text = cls._build_text(item)
                    if not text:
                        continue
                    label = int(item.get("label", item.get("class", -1)))
                    if label < 0:
                        continue
                    texts.append(text)
                    labels.append(label)
            else:
                num_labels = max(1, len(cls.LABEL_NAMES))
                target_per_label = max(1, (max_samples + num_labels - 1) // num_labels)
                label_buckets = {label: [] for label in range(num_labels)}
                for item in ds:
                    label = int(item.get("label", item.get("class", -1)))
                    if label not in label_buckets or len(label_buckets[label]) >= target_per_label:
                        continue
                    text = cls._build_text(item)
                    if not text:
                        continue
                    label_buckets[label].append(text)
                    if all(len(bucket) >= target_per_label for bucket in label_buckets.values()):
                        break

                texts = []
                labels = []
                for label in range(num_labels):
                    for sample_text in label_buckets[label]:
                        texts.append(sample_text)
                        labels.append(label)
                        if len(texts) >= max_samples:
                            break
                    if len(texts) >= max_samples:
                        break
            return _with_source(cls(texts, labels, tokenizer, max_length), "huggingface")
        except Exception as exc:
            print(f"Error loading DBPedia ({type(exc).__name__}: {exc}); fallback disabled by default.")
            _require_fallback_permission("dbpedia_14")
            return _with_source(
                cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples),
                "synthetic_fallback",
            )

class ContinualSubsetDataset(Dataset):
    """Subset wrapper for continual classification tasks."""

    def __init__(
        self,
        base_dataset: LongTextDataset,
        indices: List[int],
        label_map: Dict[int, int],
        task_id: int,
        label_mode: str = "local",
    ):
        if label_mode not in {"local", "global"}:
            raise ValueError("label_mode must be either 'local' or 'global'")
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map
        self.task_id = task_id
        self.label_mode = label_mode
        self.task_labels = sorted(label_map.keys())
        self.output_labels = (
            sorted(label_map.values()) if label_mode == "local" else self.task_labels
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        tokens, label = self.base_dataset[self.indices[idx]]
        raw_label = int(label.item())
        output_label = self.label_map[raw_label] if self.label_mode == "local" else raw_label
        return tokens, torch.tensor(output_label, dtype=torch.long), torch.tensor(self.task_id, dtype=torch.long)


def build_split_classification_tasks(
    dataset: LongTextDataset,
    classes_per_task: int = 2,
    label_mode: str = "local",
) -> List[Dataset]:
    if label_mode not in {"local", "global"}:
        raise ValueError("label_mode must be either 'local' or 'global'")
    unique_labels = sorted(set(int(label) for label in dataset.labels))
    tasks = []
    for task_id, start_index in enumerate(range(0, len(unique_labels), classes_per_task)):
        task_labels = unique_labels[start_index : start_index + classes_per_task]
        label_map = {label: mapped_index for mapped_index, label in enumerate(task_labels)}
        indices = [index for index, label in enumerate(dataset.labels) if int(label) in label_map]
        tasks.append(ContinualSubsetDataset(dataset, indices, label_map, task_id, label_mode=label_mode))
    return tasks


def _fetch_raw_texts(dataset_name: str, split: str, max_samples: Optional[int] = None) -> List[str]:
    """Fetch raw text strings for tokenizer training."""
    from datasets import load_dataset
    texts: List[str] = []
    if dataset_name == "split_ag_news":
        ds = load_dataset("ag_news", split=split)
        for item in ds:
            if "title" in item and "description" in item:
                text = f"Headline: {item.get('title','')}\n\nArticle: {item.get('description','')}"
            elif "text" in item:
                text = str(item["text"])
            else:
                text = ""
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break
    elif dataset_name == "split_dbpedia":
        ds = load_dataset("dbpedia_14", split=split)
        for item in ds:
            title = str(item.get("title") or item.get("name") or "")
            content = str(item.get("content") or item.get("text") or "")
            texts.append(f"Title: {title}\n\nContent: {content}" if title else content)
            if max_samples and len(texts) >= max_samples:
                break
    elif dataset_name == "split_arxiv":
        ds = load_dataset("scientific_papers", "arxiv", split=split)
        for item in ds:
            text = f"Title: {str(item.get('article', item.get('abstract','')))[:2000]}"
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break
    return texts


def get_continual_dataloaders(
    dataset_name: str,
    batch_size: int = 8,
    max_length: int = 2048,
    classes_per_task: int = 2,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    tokenizer=None,
    tokenizer_vocab_size: int = 10000,
    use_char_tokenizer: bool = False,
    label_mode: str = "local",
):
    if tokenizer is None:
        if use_char_tokenizer:
            tokenizer = SimpleCharTokenizer()
        else:
            tokenizer = BPETokenizer(vocab_size=tokenizer_vocab_size)
            try:
                raw_texts = _fetch_raw_texts(dataset_name, "train", max_samples=max_train_samples)
                if raw_texts:
                    tokenizer.train(raw_texts[:min(len(raw_texts), 5000)])
            except Exception:
                pass

    if dataset_name == "split_ag_news":
        train_dataset = AGNewsDataset.load(
            split="train",
            max_length=max_length,
            max_samples=max_train_samples,
            tokenizer=tokenizer,
        )
        val_dataset = AGNewsDataset.load(
            split="test",
            max_length=max_length,
            max_samples=max_val_samples,
            tokenizer=tokenizer,
        )
    elif dataset_name == "split_arxiv":
        train_dataset = ArXivDataset.load(
            split="train",
            max_length=max_length,
            max_samples=max_train_samples,
            tokenizer=tokenizer,
        )
        val_dataset = ArXivDataset.load(
            split="test",
            max_length=max_length,
            max_samples=max_val_samples,
            tokenizer=tokenizer,
        )
    elif dataset_name == "split_dbpedia":
        train_dataset = DBPediaDataset.load(
            split="train",
            max_length=max_length,
            max_samples=max_train_samples,
            tokenizer=tokenizer,
        )
        val_dataset = DBPediaDataset.load(
            split="test",
            max_length=max_length,
            max_samples=max_val_samples,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown continual dataset: {dataset_name}")

    train_tasks = build_split_classification_tasks(
        train_dataset,
        classes_per_task=classes_per_task,
        label_mode=label_mode,
    )
    val_tasks = build_split_classification_tasks(
        val_dataset,
        classes_per_task=classes_per_task,
        label_mode=label_mode,
    )

    train_loaders = [
        DataLoader(task_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        for task_dataset in train_tasks
    ]
    val_loaders = [
        DataLoader(task_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        for task_dataset in val_tasks
    ]
    return train_loaders, val_loaders


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
        except ImportError as exc:
            print(f"datasets not installed ({exc}); fallback disabled by default.")
            _require_fallback_permission("arxiv")
            return _with_source(
                cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples),
                "synthetic_fallback",
            )

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
                    return _with_source(
                        cls(texts, labels, tokenizer, max_length),
                        "huggingface",
                    )

                load_errors.append(f"{dataset_name}: no valid samples after filtering")
            except Exception as exc:
                load_errors.append(f"{dataset_name}: {exc}")

        print("Error loading ArXiv datasets; fallback disabled by default.")
        _require_fallback_permission("arxiv")
        for error in load_errors:
            print(f"  - {error}")
        return _with_source(
            cls._dummy_dataset(max_length, tokenizer, max_samples=max_samples),
            "synthetic_fallback",
        )


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
    """BPE tokenizer wrapper.

    Uses HuggingFace tokenizers for BPE, falling back to a word-level
    tokenizer built from training texts when tokenizers is unavailable.
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._trained = False
        self._use_hf = False
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
            self._hf_tokenizer = Tokenizer(models.BPE())
            self._hf_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            self._hf_trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            )
            self._use_hf = True
        except ImportError:
            self._hf_tokenizer = None
            self._hf_trainer = None

    def train(self, texts: List[str]):
        if self._use_hf and self._hf_tokenizer is not None:
            self._hf_tokenizer.train_from_iterator(texts, trainer=self._hf_trainer)
        else:
            word_freq: Dict[str, int] = {}
            for text in texts:
                for word in text.lower().split():
                    word_freq[word] = word_freq.get(word, 0) + 1
            vocab = sorted(word_freq.items(), key=lambda x: -x[1])
            vocab = vocab[:self.vocab_size - 5]
            specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            for i, tok in enumerate(specials):
                self._word_to_id[tok] = i
                self._id_to_word[i] = tok
            for i, (word, _) in enumerate(vocab):
                idx = i + len(specials)
                self._word_to_id[word] = idx
                self._id_to_word[idx] = word
        self._trained = True

    def encode(self, text: str) -> List[int]:
        if self._use_hf and self._hf_tokenizer is not None:
            return self._hf_tokenizer.encode(text).ids
        if not self._trained:
            return [ord(c) % self.vocab_size for c in text]
        result = []
        for word in text.lower().split():
            result.append(self._word_to_id.get(word, self._word_to_id.get("[UNK]", 1)))
        return result

    def decode(self, tokens: List[int]) -> str:
        if self._use_hf and self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(tokens)
        if not self._id_to_word:
            return " ".join(str(t) for t in tokens)
        return " ".join(self._id_to_word.get(t, "[UNK]") for t in tokens)

    def vocab_size_property(self) -> int:
        return self.vocab_size


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
