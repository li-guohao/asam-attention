# Efficient Sparse Attention: A Comprehensive Survey

**Author**: Guohao Li  
**Date**: January 2026  
**Purpose**: Positioning ASAM in the landscape of efficient attention mechanisms

---

## Table of Contents

1. [Introduction](#introduction)
2. [Taxonomy of Efficient Attention](#taxonomy)
3. [Fixed Pattern Methods](#fixed-pattern)
4. [Learnable Pattern Methods](#learnable-pattern)
5. [Kernel-Based Methods](#kernel-based)
6. [Recurrence-Based Methods](#recurrence-based)
7. [Memory-Efficient Methods](#memory-efficient)
8. [Comparison Table](#comparison)
9. [ASAM Positioning](#asam-positioning)
10. [Future Directions](#future-directions)

---

## 1. Introduction {#introduction}

Self-attention in Transformers has O(n²) time and memory complexity with respect to sequence length, making it a bottleneck for long sequences. This survey categorizes methods that aim to reduce this complexity while maintaining performance.

**Key Challenge**: The fundamental trade-off between computational efficiency and model expressiveness.

---

## 2. Taxonomy of Efficient Attention {#taxonomy}

```
Efficient Attention Methods
├── Pattern-Based Sparse Attention
│   ├── Fixed Patterns (Pre-defined sparsity)
│   │   ├── Local/Window Attention
│   │   ├── Strided Attention
│   │   ├── Dilated Attention
│   │   └── Block-wise Attention
│   ├── Composable Patterns (Multiple patterns combined)
│   │   ├── Local + Global
│   │   ├── Local + Strided + Random
│   │   └── Hierarchical Multi-scale
│   └── Learnable Patterns (Data-driven sparsity)
│       ├── Clustering-based
│       ├── Routing/Hashing-based
│       └── Gating-based (ASAM falls here)
├── Low-Rank/Kernel Methods
│   ├── Low-Rank Approximation (Linformer)
│   ├── Kernel Feature Maps (Performer)
│   └── Orthogonal Random Features
├── Recurrence-Based Methods
│   ├── RNN-like Attention (RWKV, RetNet)
│   ├── State Space Models (Mamba, S4)
│   └── Linear Attention with Recurrence
└── Memory-Efficient Methods
    ├── Flash Attention (IO-aware)
    ├── Gradient Checkpointing
    └── KV Cache Optimization
```

---

## 3. Fixed Pattern Methods {#fixed-pattern}

### 3.1 Local (Sliding Window) Attention

**Papers**:
- **Longformer** (Beltagy et al., 2020) - "Longformer: The Long-Document Transformer"
  - Local window + Global attention on pre-selected tokens
  - O(n×w) complexity where w is window size
  - Good for documents with local coherence

- **S4D** (Gu et al., 2022) - Related to structured masks

**Characteristics**:
- ✅ Simple, efficient
- ❌ Cannot capture long-range dependencies outside window
- ❌ Fixed pattern regardless of input

### 3.2 Strided Attention

**Papers**:
- **Sparse Transformer** (Child et al., 2019) - "Generating Long Sequences with Sparse Transformers"
  - Strided patterns for different heads
  - Factorized attention: row-wise + column-wise

**Characteristics**:
- ✅ Regular pattern, easy to optimize
- ❌ May miss important non-periodic dependencies

### 3.3 Block-wise Attention

**Papers**:
- **Blockwise Attention** (multiple variants)
- **Swin Transformer** (Liu et al., 2021) - Shifted windows for vision

---

## 4. Learnable Pattern Methods {#learnable-pattern}

### 4.1 Clustering-Based

**Papers**:
- **Reformer** (Kitaev et al., 2020) - "Reformer: The Efficient Transformer"
  - LSH (Locality Sensitive Hashing) to bucket similar queries/keys
  - O(n log n) complexity
  - ❌ Unstable, sensitive to hash quality

- **Routing Transformer** (Roy et al., 2021)
  - K-means clustering for attention clusters
  - Learnable centroids

- **Tactic** (Zhu et al., 2025) - "Adaptive Sparse Attention with Clustering and Distribution Fitting"
  - **Inference-time** clustering for KV cache optimization
  - Cumulative attention score threshold
  - Post-training, calibration-free
  - Focus: LLM decoding speedup

### 4.2 Gating/Routing-Based

**Papers**:
- **Mixture of Attention Heads** (Zhang et al., 2022)
  - Different heads use different patterns

- **ASAM** (Your Work, 2024)
  - **Training-time** adaptive gating
  - Learnable complexity estimator + confidence predictor
  - Differentiable sparse-dense switching
  - Hierarchical multi-scale patterns
  - Focus: End-to-end trainable architecture

**Key Difference from Tactic**:
| Aspect | Tactic | ASAM |
|--------|--------|------|
| Stage | Post-training inference | Training-time architecture |
| Adaptivity | To attention sparsity | To input complexity |
| Differentiability | No (post-hoc) | Yes (end-to-end) |
| Clustering use | Sort tokens by importance | Define attention patterns |
| Distribution fitting | Yes | No |
| KV cache focus | Core optimization | Not primary |

### 4.3 Content-Based Selection

**Papers**:
- **H2O (Heavy Hitter Oracle)** (Zhang et al., 2023)
  - Evict least important KV pairs based on attention scores
  - KV cache eviction policy

- **StreamingLLM** (Xiao et al., 2023)
  - Keep initial tokens + recent window
  - "Attention sink" phenomenon

- **SnapKV** (Li et al., 2024)
  - Observation window for KV cache compression

---

## 5. Kernel-Based Methods {#kernel-based}

### 5.1 Kernel Feature Maps

**Papers**:
- **Performer** (Choromanski et al., 2020) - "Rethinking Attention with Performers"
  - FAVOR+ (Fast Attention Via Orthogonal Random features)
  - Approximates softmax with random feature maps
  - O(n×m) where m is number of features
  - Unbiased estimator of full attention

- **Linear Attention** (Katharopoulos et al., 2020)
  - softmax(QK^T)V → φ(Q)(φ(K)^T V)
  - RNN-like recurrence for autoregressive generation

### 5.2 Low-Rank Approximation

**Papers**:
- **Linformer** (Wang et al., 2020) - "Linformer: Self-Attention with Linear Complexity"
  - Project keys/values to lower dimension k << n
  - O(n×k) complexity
  - Fixed projection matrices (or learned)
  - ❌ Quality degrades for long sequences

- **Nyströmformer** (Xiong et al., 2021)
  - Nyström method for attention matrix approximation

---

## 6. Recurrence-Based Methods {#recurrence-based}

### 6.1 Linear Attention with Recurrence

**Papers**:
- **RWKV** (Peng et al., 2023) - "RWKV: Reinventing RNNs for the Transformer Era"
  - Linear attention + time-mixing
  - Combines parallel training (Transformer-like) with efficient inference (RNN-like)
  - O(1) memory per step during inference

- **RetNet** (Sun et al., 2023) - "Retentive Network: A Successor to Transformer"
  - Retention mechanism with dual recurrence and parallel forms
  - Chunk-wise recurrence for training

### 6.2 State Space Models (SSMs)

**Papers**:
- **S4** (Gu et al., 2021) - "Efficiently Modeling Long Sequences with Structured State Spaces"
  - HiPPO initialization for long-range memory
  - Structured state space

- **Mamba** (Gu & Dao, 2023) - "Linear-Time Sequence Modeling with Selective State Spaces"
  - Selective SSM with input-dependent parameters
  - Hardware-aware algorithm
  - O(n) complexity

- **Mamba-2** (Dao & Gu, 2024)
  - Structured attention duality
  - SSD (State Space Duality) algorithm

---

## 7. Memory-Efficient Methods {#memory-efficient}

### 7.1 IO-Aware Attention

**Papers**:
- **Flash Attention** (Dao et al., 2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention"
  - Tiling to reduce HBM accesses
  - Recomputation of softmax statistics
  - O(n²) compute but memory-efficient
  - No approximation

- **Flash Attention-2** (Dao, 2023)
  - Better parallelism on GPU warps
  - Reduced non-matmul FLOPs

- **Flash Attention-3** (NVIDIA, 2024)
  - FP8 support, Hopper GPU optimizations

- **FlashDecoding** (Dao et al., 2023)
  - Split-KV for long context decoding

- **Ring Attention** (Liu et al., 2024)
  - Distributed attention across devices
  - Overlapping communication with computation

### 7.2 KV Cache Optimization

**Papers**:
- **vLLM** (Kwon et al., 2023) - "Efficient Memory Management for Large Language Model Serving"
  - PagedAttention: Page-based KV cache
  - Dynamic memory allocation

- **PagedAttention v2/v3** - Various improvements

- **Tactic** (Zhu et al., 2025) - Also falls here
  - Adaptive token selection for KV cache loading

---

## 8. Comprehensive Comparison Table {#comparison}

| Method | Year | Complexity | Pattern | Learning | Stage | Key Innovation |
|--------|------|------------|---------|----------|-------|----------------|
| **Sparse Transformer** | 2019 | O(n√n) | Fixed strided | No | Architecture | Factorized attention |
| **Reformer** | 2020 | O(n log n) | LSH bucketing | Yes | Architecture | Hash-based clustering |
| **Linformer** | 2020 | O(nk) | Low-rank | Optional | Architecture | Linear projection |
| **Performer** | 2020 | O(nm) | Kernel features | No | Architecture | Random feature maps |
| **Longformer** | 2020 | O(nw) | Local + Global | No | Architecture | Window + global tokens |
| **BigBird** | 2020 | O(n) | Random+Window+Global | No | Architecture | Random graph attention |
| **Linear Attention** | 2020 | O(n) | Kernel | No | Architecture | RNN-like recurrence |
| **S4** | 2021 | O(n) | State space | Yes | Architecture | HiPPO initialization |
| **Flash Attention** | 2022 | O(n²) | Exact | N/A | Implementation | IO-aware tiling |
| **H2O** | 2023 | O(n) | Eviction policy | No | Inference | Heavy hitter retention |
| **RWKV** | 2023 | O(n) | Time-mixing | Yes | Architecture | Linear + channel mix |
| **RetNet** | 2023 | O(n) | Retention | Yes | Architecture | Dual recurrence |
| **Mamba** | 2023 | O(n) | Selective SSM | Yes | Architecture | Input-dependent SSM |
| **StreamingLLM** | 2023 | O(n) | Sink + Recent | No | Inference | Attention sink |
| **Tactic** | 2025 | O(n log n) | Adaptive cluster | No | Inference | Distribution fitting |
| **ASAM** | 2024 | O(n√n) | Adaptive gate | Yes | Architecture | Differentiable gating |

---

## 9. ASAM Positioning {#asam-positioning}

### Where ASAM Fits

```
Unique Position: Training-time + Differentiable + Adaptive + Hierarchical

                Fixed Patterns          Learnable Patterns
                (Longformer,            (Reformer,
                 BigBird)                Routing,
                                         Tactic [Inference])
                      \
                       \
                        +-----> ASAM <-----+
                        /      (Ours)       \
                       /                      \
              Pattern Composable         Memory Efficient
              (Sparse Trans)             (Flash Attn)
```

### Key Differentiators

1. **Differentiable Adaptive Gating**
   - Most methods use fixed patterns or post-hoc selection
   - ASAM learns to adapt during training (gradient-based)

2. **Hierarchical Multi-Scale**
   - Combines multiple granularities (local + strided + global)
   - Learnable combination weights

3. **End-to-End Training**
   - Unlike inference-only methods (Tactic, H2O, StreamingLLM)
   - Sparsity emerges naturally from data

4. **No Calibration Required**
   - Unlike some adaptive methods that need calibration data
   - Adaptive gating is learned, not hand-tuned

### Comparison with Closest Works

| Aspect | Tactic (2025) | Reformer (2020) | ASAM (2024) |
|--------|---------------|-----------------|-------------|
| **Stage** | Post-training | Training | Training |
| **Clustering** | Sort tokens | Bucket Q/K | Define patterns |
| **Distribution Fitting** | Yes | No | No |
| **Differentiable** | No | Partial | Yes |
| **Pattern Type** | Dynamic selection | Fixed buckets | Adaptive hierarchical |
| **Gating** | Score threshold | Hash collision | Learned complexity |
| **Focus** | KV cache loading | Memory efficiency | Training efficiency |

---

## 10. Future Directions {#future-directions}

### Open Challenges

1. **Theoretical Understanding**
   - When does sparse attention approximate full attention well?
   - What are the fundamental limits?

2. **Hardware-Software Co-design**
   - Flash Attention-style optimizations for sparse patterns
   - Custom CUDA kernels for adaptive sparsity

3. **Hybrid Methods**
   - Combine ASAM's adaptive gating with Flash Attention's efficiency
   - Integration with State Space Models (Mamba)

4. **Task-Specific Adaptation**
   - Different sparsity patterns for different tasks
   - Multi-task learning with task-specific gates

5. **Interpretability**
   - Understanding what patterns the model learns
   - Visualization of attention flow

### Opportunities for ASAM

1. **Integration with Flash Attention**
   - Use Flash Attention kernels for dense path
   - Custom kernels for sparse hierarchical patterns

2. **Hybrid with SSMs**
   - Use Mamba for local dependencies
   - Use ASAM for long-range structured attention

3. **Multi-Modal Extension**
   - Vision: Hierarchical spatial attention
   - Audio: Time-frequency patterns

4. **Inference Optimization**
   - KV cache optimization for ASAM patterns
   - Speculative decoding with sparse attention

---

## References

### Key Survey Papers
1. Tay et al. (2020) - "Efficient Transformers: A Survey"
2. Tay et al. (2022) - "Efficient Transformers: A Survey" (Updated)
3. Niu et al. (2024) - "A Survey on Efficient Inference for Large Language Models"

### Foundation Papers
- Vaswani et al. (2017) - "Attention is All You Need"
- Child et al. (2019) - "Generating Long Sequences with Sparse Transformers"
- Kitaev et al. (2020) - "Reformer"
- Wang et al. (2020) - "Linformer"
- Choromanski et al. (2020) - "Performer"
- Beltagy et al. (2020) - "Longformer"
- Zaheer et al. (2020) - "Big Bird"
- Dao et al. (2022) - "Flash Attention"

### Recent Advances (2023-2025)
- Gu & Dao (2023) - "Mamba"
- Sun et al. (2023) - "RetNet"
- Peng et al. (2023) - "RWKV"
- Xiao et al. (2023) - "StreamingLLM"
- Zhu et al. (2025) - "Tactic"

---

## Appendix: Methodological Notes

### ASAM's Unique Technical Stack

```
1. Multi-Scale Feature Extraction
   └─> Multi-scale pooling for complexity estimation
   
2. Learnable Complexity Estimation
   └─> MLP-based complexity predictor per head
   
3. Confidence Prediction
   └─> Uncertainty estimation for gating
   
4. Differentiable Gating
   └─> Soft switching: g × sparse + (1-g) × dense
   
5. Hierarchical Patterns
   └─> Learnable combination of multiple scales
   
6. Clustered Sparse Pattern
   └─> Learnable centroids for semantic grouping
```

This combination is unique among existing methods.

---

*Last Updated: January 2026*
