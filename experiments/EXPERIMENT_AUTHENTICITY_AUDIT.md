# 实验真实性审计

## Verdict

本审计未发现可直接指向“伪造实验结果”的证据；现有材料更接近于实验产物老化、语义复用、manifest 可追溯性不足与代码/产物时间线不一致的组合问题。与此同时，当前仓库中的实验产物不能作为论文级结果签收，也不应被写入论文主表、摘要或结论性表述。【假设审查：未采纳“直接伪造”结论，因为审计对象中没有发现手工篡改数值、伪造日志或伪造运行命令的直接证据；但也未采纳“可背书论文结果”结论，因为 manifest provenance 缺失、suite 语义混用、且现行代码已引入旧产物未覆盖的新机制。】

| suite | rating | 审计结论 |
| --- | --- | --- |
| `paper_suite_accuracy` | MIXED | 有候选配置语义，产物较完整，但缺少可签收 provenance，且不能证明覆盖当前代码状态。 |
| `paper_suite_retention` | MIXED | 有候选配置语义，产物较完整，但缺少可签收 provenance，且存在与 `paper_suite_retention_no_transport` 的高度复用。 |
| `paper_suite_retention_no_transport` | MIXED | 有候选配置语义，并明确 `transport_weight=0.0`，但 manifest 仍缺少可签收 provenance，且与 retention suite 共享大量数值。 |
| `paper_suite` | OUTDATED | 旧命名产物，缺少候选 profile 语义与 provenance，不应作为当前论文证据。 |
| `paper_suite_actual` | OUTDATED | 旧命名产物，结果语义与后续候选 suite 不一致，缺少 provenance。 |
| `paper_suite_paperish` | OUTDATED | 旧命名产物，命名本身偏草稿语义，缺少 provenance。 |
| `paper_suite_realdata` | OUTDATED | 与 `paper_suite` 出现高度相同的报告/表格结果，缺少 provenance，不能单独支撑“真实数据最终结果”。 |
| CURRENT | 无 | 当前没有一个带完整 provenance、覆盖现行代码并经真实重跑确认的签收 suite。 |

## Evidence

- 旧 manifest 缺 provenance：`experiments/paper_suite/paper_suite_manifest.json`、`experiments/paper_suite_actual/paper_suite_manifest.json`、`experiments/paper_suite_paperish/paper_suite_manifest.json`、`experiments/paper_suite_realdata/paper_suite_manifest.json` 均只记录配置、输出路径和同步 TeX 路径，未记录 git commit、dirty 状态、运行命令、环境、开始/结束时间或数据源校验。【假设审查：未把“缺 provenance”解释为伪造；它只能证明不可追溯，不足以证明数据被捏造。】
- 新候选 manifest 仍未达到签收级 provenance：`experiments/paper_suite_accuracy/paper_suite_manifest.json`、`experiments/paper_suite_retention/paper_suite_manifest.json`、`experiments/paper_suite_retention_no_transport/paper_suite_manifest.json` 相比旧 suite 增加了 `candidate_profile`、`candidate_profile_description`、`resolved_config` 等候选语义；但当前工作树内未发现名为 `provenance` 的字段，也未发现等价的 commit/command/env 记录。因此“新 manifest 有候选语义”成立，“新 manifest 有完整 provenance”在当前文件状态下不成立，必须经后续重跑修复后才可成立。【假设审查：这里主动否决“把 candidate_profile 当作 provenance”的路径；profile 解释实验意图，不证明实验如何、何时、由何代码运行。】
- 当前代码新增或已接入 `dual_transport`：`experiments/run_continual_text_ablation.py` 将 `dual_transport` 列入 ablation strategy；`experiments/run_continual_text_benchmark.py` 包含 `build_dual_transport_gradients`、任务级 transport weight 初始化/更新与 `adaptation_strategy == "dual_transport"` 分支；`tests/test_continual_real_benchmark.py` 覆盖 dual transport 的任务级权重与遗忘信号行为。旧 suite 不足以代表该机制的当前实现状态。
- 当前代码新增或已接入 `masked_sinkhorn_topk`：`asam/continual_asam.py` 包含 `_route_with_masked_sinkhorn` 与 `routing_strategy == "masked_sinkhorn_topk"` 分支；`experiments/run_continual_operator_ablation.py` 将 `masked_sinkhorn_topk` 纳入 `OPERATOR_STRATEGIES`；`tests/test_continual_asam.py` 与 `tests/test_continual_operator_ablation.py` 覆盖该路由策略。现有 paper-facing suite 表格主要仍围绕 `sinkhorn_topk`、`kl_topk`、`no_transport` 等旧/基线策略，不能自动外推到当前新增机制。
- 跨 suite 语义复用明显：`experiments/paper_suite/paper_suite_report.md` 与 `experiments/paper_suite_realdata/paper_suite_report.md` 均报告 `Meta-secant avg accuracy: 0.5312`、`Best avg accuracy: no_adaptation (0.5000)`、`Best operator avg accuracy: sinkhorn_topk (0.5000)`；`experiments/paper_suite/continual_operator_ablation_table.md` 与 `experiments/paper_suite_realdata/continual_operator_ablation_table.md` 的主要行数值相同。`experiments/paper_suite_retention/continual_operator_ablation_table.md` 与 `experiments/paper_suite_retention_no_transport/continual_operator_ablation_table.md` 也共享同一组 operator 统计。该现象可能来自相同配置、复制产物或重跑未区分目录；无 provenance 时不能判定具体原因。【假设审查：未直接认定为复制造假，因为相同 seed/配置可能产生相同结果；但对于论文签收，语义不同的 suite 出现相同产物而无运行证明，本身已构成阻断条件。】
- 生成 manifest 的当前脚本 `scripts/run_continual_paper_suite.py` 在写入 `paper_suite_manifest.json` 时记录 `config`、`resolved_config`、候选 profile 与输出文件路径；从当前代码块看，未写入 git commit、命令行、环境或数据校验。即使重新运行，若不先补 provenance 字段，也只能得到“较新但仍不可签收”的 manifest。

## Required Fixes

1. 在 `scripts/run_continual_paper_suite.py` 生成 manifest 时增加严格 provenance：至少包括 git commit、git dirty 状态、完整 argv、Python/PyTorch/主要依赖版本、设备、开始/结束时间、数据集名称与样本上限、随机种子列表、输出文件哈希。修复后生成的新 `paper_suite_manifest.json` 才能被称为“有 provenance”的新 manifest。
2. 废弃或隔离 `paper_suite`、`paper_suite_actual`、`paper_suite_paperish`、`paper_suite_realdata` 的论文引用入口；这些目录只能作为历史参考，不得作为当前论文结果来源。
3. 对 `paper_suite_accuracy`、`paper_suite_retention`、`paper_suite_retention_no_transport` 做真实重跑，并在重跑后重新评级；在 provenance 未补齐、重跑未完成前，三者只能维持 MIXED，不能进入论文主结论。
4. 对 `dual_transport` 与 `masked_sinkhorn_topk` 分别建立明确的 suite 语义：若论文声称当前方法包含这些机制，必须有覆盖这些策略的重跑产物、表格和 manifest；若论文不声称这些机制，则需在文档中明确其为未纳入论文结果的后续代码。
5. 重跑命令建议如下；这些命令仅用于生成待审计产物，必须在真实重跑完成、manifest provenance 完整、结果表格与日志复核后，才可用于论文：

```powershell
python scripts/run_continual_paper_suite.py --output-dir experiments/paper_suite_accuracy --candidate-profile accuracy --num-seeds 3 --device cpu --paper-tex paper/asam_paper.tex --paper-output-tex paper/asam_paper_accuracy.tex --appendix-only-tex paper/continual_appendix_accuracy.tex
python scripts/run_continual_paper_suite.py --output-dir experiments/paper_suite_retention --candidate-profile retention --num-seeds 3 --device cpu --paper-tex paper/asam_paper.tex --paper-output-tex paper/asam_paper_retention.tex --appendix-only-tex paper/continual_appendix_retention.tex
python scripts/run_continual_paper_suite.py --output-dir experiments/paper_suite_retention_no_transport --candidate-profile retention_no_transport --num-seeds 3 --device cpu --paper-tex paper/asam_paper.tex --paper-output-tex paper/asam_paper_retention_no_transport.tex --appendix-only-tex paper/continual_appendix_retention_no_transport.tex
```
