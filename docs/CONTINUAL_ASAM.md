# Continual ASAM

`ContinualASAMLayer` is a minimal continual-learning extension of ASAM that turns sparse attention into a task-separating mechanism.

## Core idea

For each task `t`, a task-aware sparse gate predicts per-head pattern weights over a shared sparse pattern bank:

- `local`
- `strided`
- `random`
- `hierarchical`

For head `h`, the layer selects the top-`k` patterns and forms a task-specific sparse support:

`M_t^{(h)} = OR_{p in TopK(w_t^{(h)})} M_p^{(h)}`

Attention is then computed only on the selected support:

`A_t^{(h)}(Q, K, V) = softmax((QK^T / sqrt(d)) masked by M_t^{(h)}) V`

Because different tasks can use different sparse supports, parameter updates are less likely to collide on the same attention edges.

## Continual-learning regularizers

The layer exposes two regularizers through `return_info=True`:

### 1. Support overlap regularization

Let `S_t` be the soft support induced by the per-head pattern weights. The overlap penalty compares the current task against both:

- other tasks present in the same batch,
- remembered sparse supports from previously seen tasks.

In both cases, the penalty has the same form:

`L_overlap = mean(S_a * S_b)`

Lower overlap encourages task-specific sparse supports and reduces interference.

### 2. Head-importance stability regularization

The task-aware gate also predicts head importances `g_t`. The layer maintains an EMA memory of previous head importances per task and computes:

`L_stability = ||g_t - g_t^{memory}||^2`

This encourages previously important heads for a revisited task to remain stable.

## Minimal usage

```python
import torch
from asam import ContinualASAMConfig, ContinualASAMLayer

config = ContinualASAMConfig(
    dim=256,
    num_heads=4,
    dim_head=64,
    num_tasks=10,
    top_k_patterns=2,
)

layer = ContinualASAMLayer(config)
x = torch.randn(8, 128, 256)
task_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

output, info = layer(x, task_ids=task_ids, return_info=True)
loss = task_loss + 0.1 * info["overlap_loss"] + 0.1 * info["stability_loss"]

layer.update_task_memory(task_ids, info["head_importance"])
```

## Current scope

This is a minimal research scaffold, not a full continual-learning training framework. It is intended to make the continual-learning math explicit and provide a runnable implementation for future experiments.

## Prototype routing

The repository also includes `PrototypeContinualASAMLayer`, a task-agnostic variant that replaces explicit `task_ids` with learned prototype routing.

For an input sequence representation `x`, the layer now computes a memory-aware sparse routing distribution over `P` learned prototypes in two stages:

`r_tilde = q_old(x) ? exp(sim(f(x), C) / tau)`

`Pi* = Sinkhorn(r_tilde, b)`

`r* = Proj_top-k(Pi*)`

where `C` is the prototype bank, `tau` is the routing temperature, `q_old` is a remembered prototype prior derived from usage statistics, `b` is a prototype-capacity target, `Sinkhorn` solves an entropic optimal-transport problem over the current batch, and `Proj_top-k` keeps only the strongest prototype assignments before renormalization.

This means continual learning is enforced directly in routing space twice: first through the KL-proximal prior bias, then through a capacity-constrained transport step that discourages prototype collapse. The routed prototype mixture is then used to predict sparse pattern weights and head importances.

Continual regularization is applied against remembered prototype statistics instead of remembered task ids, and the prototype variant also adds a routing-stability term against the remembered prior when prototype memories already exist. The balance term is now measured against the Sinkhorn capacity target instead of a fixed uniform distribution.

To reduce prototype collapse, the prototype-routed variant also exposes two extra regularizers:

- `balance_loss`: encourages the average routing distribution to stay close to uniform,
- `diversity_loss`: penalizes similarity between prototype embeddings.

It also supports an OT-driven prototype lifecycle:

- overloaded prototypes are identified by positive transport excess `usage - capacity`,
- underfilled prototypes are identified by capacity deficits and low support rates,
- split operations relocate deficit slots toward transport barycenters derived from overloaded prototype latents,
- reset operations only recycle prototypes that are simultaneously low-usage, low-capacity, and low-support, and they are reinitialized toward the current deficit barycenter instead of pure random noise.

## Mathematical grounding

This section rewrites the continual-learning argument in an appendix-style form. The goal is not to claim a fully assumption-free forgetting theorem, but to state clearly which quantities are controlled, what assumptions are needed, and how the implementation realizes those quantities.

### Notation

Let `z(x) in R^d` be the normalized latent representation of an input sequence, let `{a_i}_{i=1}^P` be normalized prototype anchors, let `p(x) in Delta^(P-1)` be the sparse routing weights, and define the routed barycenter

`mu(x) = sum_i p_i(x) a_i`

Let the average prototype usage be `u = E[p(x)]`, let the capacity target be `c in Delta^(P-1)`, and define the excess vector

`e = u - c`

A useful surrogate objective for continual sparse routing is

`Psi = lambda_T E[T(x)] + lambda_R E[R(x)] + lambda_E ||e||_2^2 + lambda_D D(A)`

where

- `T(x) = sum_i p_i(x) (1 - z(x)^T a_i)` is the transport term,
- `R(x) = KL(p(x) || r(x))` is the routing-stability term against remembered prior `r(x)`,
- `D(A) = sum_{i != j} <a_i, a_j>^2` is the prototype-diversity penalty.

### Assumption 1 (Normalized geometry)

Assume `||z(x)||_2 = 1` and `||a_i||_2 = 1` for all samples and prototypes. This is the regime implemented by the normalized prototype router.

### Assumption 2 (Stable memory prior)

Assume the remembered prior `r(x)` approximates the old routing distribution `p_old(x)` up to memory error `delta_mem`, i.e.

`||r(x) - p_old(x)||_1 <= delta_mem`

### Assumption 3 (Lipschitz downstream head)

Assume the downstream classifier or task head is `L`-Lipschitz with respect to the routed barycenter `mu(x)`.

### Lemma 1 (Entropic OT interpretation of the router)

Define the batch-to-prototype cost matrix

`C_{bi} = 1 - z_b^T a_i - beta log r_{bi}`

Then the dense routing plan before top-`k` projection can be written as the solution of the entropic optimal-transport problem

`Q* = argmin_{Q >= 0} <C, Q> + eps sum_{b,i} Q_{bi} (log Q_{bi} - 1)`

subject to

`Q 1 = (1 / B) 1,    Q^T 1 = c`

with `c` the prototype-capacity target. Therefore the current Sinkhorn router is a capacity-constrained OT assignment, not merely a softmax heuristic.

### Lemma 2 (Exact error of top-`k` projection)

Let `Pi_k(p)` be the renormalized top-`k` projection of a routing vector `p`, and let the discarded tail mass be

`m = 1 - sum_{i in TopK(p)} p_i`

Then

`||p - Pi_k(p)||_1 = 2m`

So the support-only sparse approximation is exact up to twice the discarded tail mass of the dense OT plan.

### Proposition 1 (Transport controls barycenter drift)

Under Assumption 1,

`1 - z^T a_i = (1/2) ||z - a_i||_2^2`

hence

`T(x) = (1/2) sum_i p_i(x) ||z(x) - a_i||_2^2`

By convexity of the squared norm,

`||z(x) - mu(x)||_2^2 <= sum_i p_i(x) ||z(x) - a_i||_2^2 = 2 T(x)`

and therefore

`||z(x) - mu(x)||_2 <= sqrt(2 T(x))`

Interpretation: a small transport loss prevents the routed prototype barycenter from drifting far from the latent geometry.

### Proposition 2 (Routing KL controls route drift)

By Pinsker's inequality,

`||p(x) - r(x)||_1 <= sqrt(2 KL(p(x) || r(x)))`

Combining this with Assumption 2 gives

`||p(x) - p_old(x)||_1 <= sqrt(2 KL(p(x) || r(x))) + delta_mem`

So the routing-stability term bounds route drift relative to remembered behavior, up to the quality of the memory prior.

### Proposition 3 (Excess is an exact interference statistic)

For two independent samples `x, x'`, define the prototype collision statistic

`I = E[p(x)^T p(x')]`

Then

`I = ||u||_2^2`

If the capacity target is uniform, `c = (1 / P) 1`, then because `sum_i e_i = 0`,

`I = ||c + e||_2^2 = ||c||_2^2 + ||e||_2^2 = 1/P + ||e||_2^2`

This identity shows that `||e||_2^2` measures routing concentration above the uniform-collision floor. Reducing excess directly reduces expected prototype interference.

### Corollary 1 (Split yields strict local descent on imbalance)

Suppose prototype `i` is overloaded with `e_i > 0` and prototype `j` is underfilled with `e_j < 0`. If a split transfers mass `alpha` from `i` to `j`, then

`e_i' = e_i - alpha,    e_j' = e_j + alpha`

and

`||e'||_2^2 - ||e||_2^2 = -2 alpha (e_i - e_j) + 2 alpha^2`

Therefore whenever `0 < alpha < e_i - e_j`, the excess energy strictly decreases. This gives the transport-aware split rule a strict local-descent guarantee on the imbalance surrogate.

### Proposition 4 (Merge-plus-recycle is a conditional local-descent step)

Suppose two prototypes `a_i` and `a_j` are redundant in the sense that `||a_i - a_j||_2 <= eps`, and that their recycled slot is relocated toward a deficit barycenter `b`.

For the mass already served by `a_i` or `a_j`, the induced transport perturbation is `O(eps)`. If the recycled slot gives transport gain `G_b` on the deficit region, then the net transport change satisfies

`Delta T <= M_ij eps - G_b`

where `M_ij` is the mass carried by the redundant pair. Thus merge-plus-recycle is beneficial whenever `G_b > M_ij eps`. Unlike split, merge is a conditional local-descent step rather than an unconditional one.

### Theorem sketch (Continual-forgetting surrogate bound)

Under Assumptions 1-3, and assuming old-task loss drift is monotone in barycenter drift and routing collision, the old-task forgetting can be upper-bounded by a surrogate of the form

`F_t <= A E[sqrt(T_t(x))] + B E[sqrt(R_t(x))] + C ||e_t||_2^2 + B delta_mem`

for task-independent constants `A, B, C`.

Proof idea:

- Proposition 1 converts transport loss into a barycenter-drift bound,
- Proposition 2 converts routing KL into a route-drift bound,
- Proposition 3 converts excess into an interference statistic,
- Corollary 1 and Proposition 4 explain why the lifecycle operators can decrease the surrogate locally.

This is still a theorem sketch rather than a full generalization theorem, but it is enough to justify the present algorithm as a mathematically coherent continual-learning surrogate with explicit control variables.

### Theory-to-implementation map

The table below makes the surrogate quantities explicit in code so that theory, training, and diagnostics all refer to the same objects.

| Theory quantity | Meaning | Implementation handle | Export / usage |
| --- | --- | --- | --- |
| `T(x)` | transport surrogate between latent and prototype anchors | `info["transport_loss"]` from `PrototypeContinualASAMLayer.forward` | added to training loss in both synthetic and real benchmarks; exported in `stage_training_metrics` and `theory_diagnostics["stage_transport_loss"]` |
| `R(x)` | routing drift against remembered prior | `info["routing_stability_loss"]` | added to benchmark diagnostics and correlations against forgetting |
| `e = u - c` | prototype imbalance / excess | `prototype_excess_ema` and lifecycle statistic `mean_excess` | exported through prototype diagnostics and lifecycle summaries |
| `D(A)` | prototype redundancy penalty | `info["diversity_loss"]` | added to training loss when `diversity_weight > 0` |
| split descent | move mass from overloaded to underfilled prototypes | `refresh_prototypes(...)["split_count"]` | exported per stage through `prototype_lifecycle` |
| merge descent | collapse redundant prototypes and recycle capacity | `refresh_prototypes(...)["merge_count"]` | exported per stage through `prototype_lifecycle` and visualized in lifecycle plots |

In code terms, the surrogate described above is now closed-loop:

- the layer computes `transport_loss`, `routing_stability_loss`, `balance_loss`, and `diversity_loss`,
- the training scaffolds actually optimize those signals,
- the benchmark exports the same signals stage-by-stage,
- the online controller adapts hyperparameters from the exported stage diagnostics instead of from unrelated proxies.

This alignment matters for rigor: the mathematical quantities used in the assumptions, lemmas, propositions, and theorem sketch are no longer only explanatory notation, they correspond to named tensors or exported metrics in the runnable implementation.
## Related work and claimed contributions

This section is intentionally conservative. It separates ingredients that are clearly present in prior work from the narrower contribution claim that is more defensible for the current repository.

### Closest prior directions

- **Routing for continual learning.** Routing Networks already studied task-conditioned path selection to reduce interference across tasks; see [Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning](https://arxiv.org/abs/1711.01239) and [Routing Networks with Co-training for Continual Learning](https://arxiv.org/abs/2009.04381).
- **Sparse attention via content-dependent routing.** Sparse/content-clustered attention predates this repository; a canonical reference is [Routing Transformer](https://arxiv.org/abs/2003.05997).
- **Optimal transport and differentiable top-k routing.** The Sinkhorn / OT view of sparse assignment is also not new by itself; representative references include [Differentiable Top-k Operator with Optimal Transport](https://arxiv.org/abs/2002.06504) and [Sparsity-Constrained Optimal Transport](https://openreview.net/forum?id=yHY9NbQJ5BP).
- **Prototype-based continual learning.** Prototype memories and replay-style prototype summaries also have prior art, for example [Variational Prototype Replays for Continual Learning](https://arxiv.org/abs/1905.09447).
- **Load balancing and routing specialization in sparse expert systems.** Capacity balancing and routing specialization are heavily studied in MoE-style systems; one useful anchor is [Training Mixture-of-Experts: A Focus on Expert-Token Matching](https://openreview.net/forum?id=3EoWeMlyJr).

### What this repository does **not** claim as novel

- It does **not** claim to invent sparse attention, top-`k` routing, Sinkhorn routing, prototype memories, or continual-learning replay in isolation.
- It does **not** yet claim a new assumption-free forgetting theorem or a closed-form convergence theorem for the full lifecycle controller.
- It does **not** claim that merge/split lifecycle control is unprecedented in the broader continual-learning or adaptive-mixture literature.

### Narrower contribution claim

The more defensible claim is a **framework-level contribution**:

> a continual sparse-attention controller that couples prototype routing, capacity-constrained OT assignment, transport-aware prototype lifecycle updates, and stage-wise continual-learning diagnostics inside one runnable ASAM implementation.

More concretely, the repository contributes the following combination:

1. **Prototype-routed sparse attention as the control surface.** Sparse attention-pattern selection is no longer only task-conditioned; it is mediated by prototype routing and therefore becomes analyzable through routing geometry.
2. **Capacity-constrained OT routing as a continual-learning mechanism.** The Sinkhorn step is used not only for balanced assignment, but as a way to define prototype capacity targets and transport excess/deficit statistics that can be tracked across tasks.
3. **Transport-aware lifecycle control.** Prototype split, merge, reset, and relocation are driven by latent barycenters, support statistics, and excess variables rather than only by ad-hoc reinitialization rules.
4. **A closed diagnostic loop.** The same quantities used in the mathematical argument (`transport_loss`, routing KL, excess, merge/split counts) are exposed as tensors, optimized in training, exported in benchmarks, and reused by the online hyperparameter controller.

### How to phrase the contribution in a paper

A careful paper-style contribution statement would be:

> We present a continual-learning extension of sparse attention in which prototype routing, capacity-constrained optimal transport, and transport-aware prototype lifecycle control are unified into a single sparse-attention framework. Our main contribution is not a new stand-alone OT operator or a new stand-alone continual-learning replay method, but the integration of these mechanisms into a coherent routing geometry with explicit continual-learning diagnostics and adaptive control signals.

A stronger version that is still reasonably cautious would be:

> To our knowledge, the present combination of prototype-routed sparse attention, Sinkhorn capacity targets, transport-excess lifecycle updates, and stage-wise continual-learning diagnostics has not been presented as a single runnable ASAM-style continual-learning system.

### Reviewer-facing positioning

If this work is written up as a paper, the safest positioning is:

- **Novelty type:** systems/method integration with mathematically motivated control variables,
- **Primary contribution:** a new continual sparse-routing formulation and lifecycle controller for ASAM,
- **Secondary contribution:** an executable diagnostic pipeline that ties transport, routing drift, and imbalance to forgetting measurements,
- **Non-claim:** not a claim of inventing OT routing, prototypes, or sparse attention independently.

This positioning is stronger than saying the method is purely engineering, but more defensible than claiming a wholly new fundamental operator.
## Training scaffold

The repository also includes a minimal sequential-task training script with synthetic task streams and a small replay buffer:

```bash
python experiments/train_continual_asam.py --num-tasks 3 --epochs-per-task 1
```

For prototype-routed continual learning:

```bash
python experiments/train_continual_asam.py --routing-mode prototype --balance-weight 0.05 --diversity-weight 0.05
```

To enable prototype lifecycle management during training:

```bash
python experiments/train_continual_asam.py --routing-mode prototype --prototype-reset-threshold 0.01 --prototype-split-threshold 0.20
```

The training scaffold now also exports prototype diagnostics after each task stage, including:

- `task_prototype_heatmap`: average prototype occupancy per seen task,
- `task_routing_entropy`: average routing entropy per seen task,
- `layer_similarity`: prototype similarity matrices,
- `layer_usage_ema`: running prototype usage statistics,
- `layer_capacity_ema`: running Sinkhorn capacity targets,
- `layer_support_ema`: running sparse-support occupancy rates,
- `layer_excess_ema`: running transport excess / deficit statistics,
- `layer_latent_ema`: prototype latent barycenters used for relocation.
- `stage_training_metrics`: per-stage loss decomposition including `transport_loss` and routing-stability signals.

## Real benchmark

For a real-text class-incremental benchmark, the repository includes a `Split AG News` runner:

```bash
python experiments/run_continual_text_benchmark.py --dataset-name split_ag_news --routing-mode prototype
```

If the Hugging Face `datasets` package or AG News download is unavailable, the loader falls back to deterministic AG-News-like samples so the pipeline remains runnable.

When `--output-json` is provided, the runner now exports a compact evaluation bundle next to the JSON file:

- `<name>.json`: raw metrics, accuracy matrix, lifecycle state, and prototype diagnostics,
- `<name>_plots.png`: continual accuracy and routing/prototype visualizations,
- `<name>_report.md`: a short Markdown summary for experiment tracking.

This makes the real benchmark more useful for continual-learning analysis because support reuse, routing entropy, and prototype occupancy can be inspected together with forgetting metrics.

The real benchmark now also exports stage-wise theory diagnostics, including forgetting traces and correlations against routing stability, transport gap, and prototype excess statistics. This helps check whether the continual-learning geometry is actually predictive of forgetting instead of only being visually plausible.

When prototype routing is enabled, the benchmark can now also adapt `prototype_prior_strength`, `prototype_capacity_blend`, and `prototype_relocation_strength` online from those diagnostics. The exported final hyperparameter state also includes the prototype merge controls, so the lifecycle configuration is reproducible instead of being partially implicit. The default strategy is now an online secant-style meta update: it compares stage-wise meta-objective changes against the previous hyperparameter move to estimate a one-step hypergradient, then mixes that estimate with the earlier correlation-based controller for stability. In other words, the continual-learning geometry is no longer only monitored; it can also feed back into the sparse routing/controller hyperparameters for subsequent tasks.

For side-by-side comparisons, the repository now also includes `experiments/run_continual_text_ablation.py`, which runs `task_routing`, `no_adaptation`, `correlation`, and `meta_secant` under the same benchmark settings across multiple seeds and exports aggregated JSON, CSV, PNG, Markdown table, and Markdown report artifacts with mean?std summaries. This is the quickest way to see whether the extra continual-learning mathematics actually improves forgetting rather than just adding diagnostics.

The replay buffer intentionally mixes old-task samples into current-task batches, but `overlap_loss` also remains active without replay because the layer now compares the current support against remembered supports from seen tasks.

## One-command pipeline

Use `scripts/run_continual_paper_suite.py` or `scripts/run_paper_continual_suite.ps1` to run the benchmark, multi-seed ablation, and final paper-ready summary in one command.
