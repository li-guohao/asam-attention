#!/usr/bin/env python3
"""
Profile sparse pattern construction, caching, and effective runtime access.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from asam.sparse_patterns import (
    HierarchicalSparsePattern,
    LocalSparsePattern,
    RandomSparsePattern,
    StridedSparsePattern,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ASAM sparse pattern performance")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num-random", type=int, default=64, help="Random connections per token")
    parser.add_argument("--repeats", type=int, default=10, help="Benchmark repeats")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Comma-separated device list: auto,cpu,cuda",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="local,strided,random,hierarchical",
        help="Comma-separated pattern list",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def resolve_devices(device_arg: str) -> List[torch.device]:
    if device_arg == "auto":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

    devices = []
    for item in device_arg.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "cuda" and not torch.cuda.is_available():
            continue
        devices.append(torch.device(item))
    return devices or [torch.device("cpu")]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
    warmup: int,
) -> Tuple[float, torch.Tensor]:
    result = None
    for _ in range(warmup):
        result = fn()
    sync_device(device)

    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    sync_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / repeats
    return elapsed_ms, result


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    return {
        "memory_mb": tensor.numel() * tensor.element_size() / (1024 ** 2),
        "density": tensor.float().mean().item(),
        "sparsity": 1.0 - tensor.float().mean().item(),
    }


def make_pattern_factories(seq_len: int, num_heads: int, num_random: int) -> Dict[str, Callable[[], torch.nn.Module]]:
    return {
        "local": lambda: LocalSparsePattern(seq_len=seq_len, window_size=min(128, seq_len)),
        "strided": lambda: StridedSparsePattern(
            seq_len=seq_len,
            stride=max(1, min(32, seq_len)),
            local_window=max(1, min(16, seq_len // 2)),
        ),
        "random": lambda: RandomSparsePattern(
            seq_len=seq_len,
            num_random=min(num_random, seq_len),
            num_heads=num_heads,
        ),
        "hierarchical": lambda: HierarchicalSparsePattern(
            seq_len=seq_len,
            scales=[4, 16, 64],
            num_heads=num_heads,
        ),
    }


def profile_pattern(
    name: str,
    factory: Callable[[], torch.nn.Module],
    device: torch.device,
    repeats: int,
    warmup: int,
) -> Dict[str, object]:
    build_ms, built_tensor = time_call(lambda: factory().build_pattern(), torch.device("cpu"), repeats, warmup)

    pattern = factory()
    first_get_ms, first_tensor = time_call(lambda: pattern.get_pattern(device), device, 1, 0)
    cached_get_ms, cached_tensor = time_call(lambda: pattern.get_pattern(device), device, repeats, warmup)

    effective_ms: Optional[float] = None
    effective_tensor = cached_tensor
    if hasattr(pattern, "combine_patterns"):
        effective_ms, effective_tensor = time_call(
            lambda: pattern.combine_patterns(device),
            device,
            repeats,
            warmup,
        )

    stats = tensor_stats(effective_tensor)
    return {
        "pattern": name,
        "device": str(device),
        "shape": list(effective_tensor.shape),
        "build_ms": build_ms,
        "first_get_ms": first_get_ms,
        "cached_get_ms": cached_get_ms,
        "effective_ms": effective_ms,
        "memory_mb": stats["memory_mb"],
        "density": stats["density"],
        "sparsity": stats["sparsity"],
        "cache_reused": bool(first_tensor.data_ptr() == cached_tensor.data_ptr()),
        "build_shape": list(built_tensor.shape),
    }


def print_table(results: List[Dict[str, object]]) -> None:
    print("=" * 120)
    print("ASAM Sparse Pattern Profile")
    print("=" * 120)
    header = (
        f"{'Pattern':<14} {'Device':<8} {'Build (ms)':>11} {'First get':>11} "
        f"{'Cached get':>11} {'Effective':>11} {'Memory':>10} {'Sparsity':>10} {'Cache':>7}"
    )
    print(header)
    print("-" * len(header))

    for item in results:
        effective_ms = "-" if item["effective_ms"] is None else f"{item['effective_ms']:.3f}"
        print(
            f"{item['pattern']:<14} {item['device']:<8} {item['build_ms']:>11.3f} "
            f"{item['first_get_ms']:>11.3f} {item['cached_get_ms']:>11.3f} {effective_ms:>11} "
            f"{item['memory_mb']:>9.3f}M {item['sparsity']:>9.2%} {str(item['cache_reused']):>7}"
        )


def main() -> None:
    args = parse_args()
    pattern_names = [item.strip().lower() for item in args.patterns.split(",") if item.strip()]
    factories = make_pattern_factories(args.seq_len, args.num_heads, args.num_random)
    devices = resolve_devices(args.devices)

    results: List[Dict[str, object]] = []
    for device in devices:
        for pattern_name in pattern_names:
            if pattern_name not in factories:
                raise ValueError(f"Unknown pattern: {pattern_name}")
            results.append(
                profile_pattern(
                    pattern_name,
                    factories[pattern_name],
                    device,
                    args.repeats,
                    args.warmup,
                )
            )

    print_table(results)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {output_path}")


if __name__ == "__main__":
    main()
