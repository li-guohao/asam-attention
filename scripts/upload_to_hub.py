#!/usr/bin/env python3
"""Upload a trained ASAM checkpoint to HuggingFace Hub.

Usage:
    python scripts/upload_to_hub.py --checkpoint checkpoints/asam-imdb-v1 --repo li-guohao/asam-imdb

Requires:
    pip install huggingface_hub
    HF token set via: huggingface-cli login
"""

import argparse

from asam.modeling_asam import ASAMHFForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Upload ASAM model to HF Hub")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to local checkpoint directory"
    )
    parser.add_argument(
        "--repo", type=str, required=True, help="HF Hub repository name (e.g. li-guohao/asam-imdb)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Create as private repository"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = ASAMHFForSequenceClassification.from_pretrained(args.checkpoint)

    print(f"Pushing to {args.repo} (private={args.private})...")
    model.push_to_hub(args.repo, private=args.private)

    print(f"Done! Model available at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
