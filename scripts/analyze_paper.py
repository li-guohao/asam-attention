#!/usr/bin/env python3
"""Analyze the Tactic paper and compare with ASAM"""

import fitz
import re

pdf_path = r'C:\Users\Administrator\Downloads\2502.12216v1.pdf'

def clean_text(text):
    """Clean text for output"""
    # Remove problematic unicode characters
    return text.encode('ascii', 'ignore').decode('ascii')

def main():
    doc = fitz.open(pdf_path)
    print(f"Paper: Tactic - Adaptive Sparse Attention with Clustering and Distribution Fitting")
    print(f"Pages: {len(doc)}")
    print("="*70)
    
    # Extract abstract
    abstract = doc[0].get_text()
    print("\n1. PAPER SUMMARY:")
    print("-" * 50)
    
    # Get key sentences from abstract
    key_points = [
        "propose Tactic",
        "sparsity-adaptive",
        "calibration-free",
        "cumulative attention scores",
        "target fraction",
        "clustering-based sorting",
        "distribution fitting",
        "7.29x decode attention speedup"
    ]
    
    for point in key_points:
        if point.lower() in abstract.lower():
            print(f"  - {point}")
    
    # Extract method details
    print("\n2. KEY TECHNICAL DETAILS (from Method section):")
    print("-" * 50)
    
    method_text = ""
    for i in range(2, min(7, len(doc))):
        method_text += doc[i].get_text()
    
    # Look for specific technical terms
    technical_terms = [
        ("cumulative attention", "Target fraction of attention scores"),
        ("token budget", "Fixed vs adaptive selection"),
        ("clustering", "K-means or similar for sorting"),
        ("distribution fitting", "Statistical approximation"),
        ("KV cache", "Inference optimization focus"),
        ("decoding", "Generation phase optimization"),
    ]
    
    for term, desc in technical_terms:
        if term.lower() in method_text.lower():
            print(f"  - {term}: {desc}")
    
    # Look for experiments
    print("\n3. EXPERIMENTS:")
    print("-" * 50)
    
    exp_text = ""
    for i in range(6, min(12, len(doc))):
        exp_text += doc[i].get_text()
    
    datasets = ["LongBench", "RULER", "Needle", "passkey", "PG-19"]
    models = ["Llama", "Mistral", "Qwen"]
    
    for dataset in datasets:
        if dataset.lower() in exp_text.lower():
            print(f"  - Dataset: {dataset}")
    
    for model in models:
        if model.lower() in exp_text.lower():
            print(f"  - Model: {model}")
    
    print("\n4. COMPARISON WITH ASAM:")
    print("="*70)
    print("\nTactic (This Paper):")
    print("  - Focus: Inference optimization (KV cache, decoding)")
    print("  - Method: Cumulative attention score threshold")
    print("  - Key: Clustering + Distribution fitting for selection")
    print("  - Adaptive: Yes (to attention sparsity)")
    print("  - Calibration: None needed")
    print("  - Application: Post-training LLM inference")
    
    print("\nASAM (Your Implementation):")
    print("  - Focus: Training-time attention mechanism")
    print("  - Method: Learnable gating between sparse/dense")
    print("  - Key: Multi-scale patterns + Learnable clustering")
    print("  - Adaptive: Yes (to input complexity)")
    print("  - Differentiable: Yes (end-to-end training)")
    print("  - Application: General sequence modeling")
    
    print("\n5. KEY DIFFERENCES:")
    print("-" * 50)
    print("  1. Stage:")
    print("     - Tactic: Post-training inference optimization")
    print("     - ASAM: Training-time architecture component")
    
    print("\n  2. Selection Criteria:")
    print("     - Tactic: Cumulative attention score threshold")
    print("     - ASAM: Learnable gate based on complexity")
    
    print("\n  3. Clustering:")
    print("     - Tactic: For sorting tokens by importance")
    print("     - ASAM: For defining sparse attention pattern")
    
    print("\n  4. Distribution Fitting:")
    print("     - Tactic: YES - Approximate token selection")
    print("     - ASAM: NO - Direct pattern-based selection")
    
    print("\n  5. KV Cache Focus:")
    print("     - Tactic: YES - Optimizes loading during decoding")
    print("     - ASAM: NO - General attention mechanism")
    
    print("\n6. CONCLUSION:")
    print("="*70)
    print("These are DIFFERENT methods with different goals:")
    print("  - Tactic: Inference-speed optimization for deployed LLMs")
    print("  - ASAM: Training-efficient architecture for long sequences")
    print("\nThey share 'adaptive sparse attention' concept but implement")
    print("it differently for different use cases.")
    
    doc.close()

if __name__ == "__main__":
    main()
