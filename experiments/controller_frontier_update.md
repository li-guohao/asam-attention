# Controller Frontier Update

## What changed

- `meta_secant` now adapts `transport_weight` online instead of only the prototype geometry scalars.
- `meta_secant` now also adapts the discrete sparse support budget `prototype_top_k` through a projected integer update.
- The secant branch is now restricted to continuous geometry parameters; bounded/discrete controls (`transport_weight`, `prototype_top_k`) use the stable bootstrap controller only.
- The continual text loader now supports `split_arxiv`, which gives a 4-task continual stream from the existing `ArXivDataset` fallback/loader.

## Verified behavior

- In unit tests, `meta_secant` reduces `transport_weight` and shrinks `prototype_top_k`, while `correlation` leaves them unchanged.
- Full regression passes: `74 passed, 2 warnings`.

## Empirical outcome

### 2-task AG News (`top_k=2`, 3 seeds)
- `meta_secant` really changes the controller state: it drives `prototype_top_k` from `2 -> 1` and `transport_weight` close to `0` on every seed.
- But the aggregate benchmark metrics remain tied with `no_adaptation` / `correlation` on this 2-task setup.
- Interpretation: the controller is no longer a no-op, but this benchmark is too short-horizon to convert the changed routing policy into a stable mean gain.

### 4-task ArXiv fallback (`split_arxiv`, 2 seeds)
- `meta_secant` now consistently converges to `prototype_top_k = 1` and `transport_weight = 0.0` by the end of the stream.
- Aggregate accuracy/forgetting still match `no_adaptation` in the fallback data regime.
- Interpretation: the adaptive controller is structurally active and stable now, but the fallback task stream is not sensitive enough to separate strategies.

## Current bottleneck

- The main remaining blocker is now benchmark sensitivity, not controller inactivity.
- We finally have a controller that actually changes the sparse operator over time; what we still lack is a multi-stage real-data stream where those operator changes measurably affect continual metrics.

## Best next move

- Run the same 4-task controller study on a real multi-class text dataset (not the fallback stream), or add another continual dataset with more than 2 tasks and stronger cross-task interference.
