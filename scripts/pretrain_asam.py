#!/usr/bin/env python3
"""Train an ASAM model on IMDB long document classification.

This script trains ASAMHFForSequenceClassification on IMDB reviews
and saves a checkpoint suitable for upload to HuggingFace Hub.

Usage:
    python scripts/pretrain_asam.py --output checkpoints/asam-imdb-v1

Requirements:
    pip install transformers datasets tqdm
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ASAM on IMDB")
    parser.add_argument("--output", type=str, default="checkpoints/asam-imdb-v1")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Output: {args.output}")

    # Load IMDB dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.with_format("torch")

    train_loader = DataLoader(
        tokenized["train"], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        tokenized["test"], batch_size=args.batch_size, shuffle=False
    )

    # Create model
    config = ASAMHFConfig(
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_labels=2,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_length,
        pad_token_id=tokenizer.pad_token_id or 0,
        pattern_type="hierarchical",
    )
    model = ASAMHFForSequenceClassification(config).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            output.loss.backward()
            optimizer.step()

            train_loss += output.loss.item()
            pbar.set_postfix(loss=f"{output.loss.item():.4f}")

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["label"].to(args.device)

                output = model(input_ids, attention_mask=attention_mask)
                preds = output.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output)
            config.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  Saved best checkpoint (acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
