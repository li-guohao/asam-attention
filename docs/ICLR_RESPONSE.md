# ICLR 审稿回复（R1）— Continual ASAM

## 总述

感谢审稿意见。本轮修订把论文定位为"持续稀疏路由的诊断框架与可复现评测套件"：我们不再主张 prototype routing 降低遗忘，而是主张 (i) 同一组路由几何变量同时用于优化、生命周期与导出诊断；(ii) 每个表格数字可回溯到带 config 快照的权威产物并由一致性测试守卫；(iii) 对诊断量在当前规模下何时有效、何时无效给出诚实刻画。以下逐条回应 Critical 意见，全部修改已提交并推送（master: `b529eef`）。

## W1 摘要拼接两套实验、"connect diagnostics to forgetting"与正文矛盾

已修复。摘要删除"task-conditioned +0.0625 → prototype −0.0938 → KL-top-k 0.4844→0.5156"的拼接链（该链把 `continual_ablation.json` 与 `continual_operator_ablation.json` 两套实验连成因果叙事），改为只引用表1 可回溯数字（BPE 3-epoch：遗忘 0.0000 vs +0.0417，准确率 0.5104 vs 0.5312）并明确标注"差异在 seed 噪声内"；末句"connect routing geometry diagnostics to forgetting"改为"导出阶段轨迹供监测，相关分析留作未来工作"。新增守卫测试 `test_abstract_matches_canonical_numbers` 断言摘要数字与 `r2_agnews_bpe_3ep.json` 一致、拼接表述不再出现（commit `b1c61b1`）。

## W2 评测处于"不产生遗忘"的体制，核心主张无从验证

承认，并已在文中显式处理：新增 "Benchmark regime" 局限条目，说明双任务 task-incremental multi-head + per-task heads + oracle task id 下所有方法遗忘≈0，class-incremental single-head smoke 接近随机；"prototype 降低遗忘"降级为初步观察而非主张。该体制下无法证明路由机制降低遗忘，验证它需要 T≥5 或 class-incremental/task-agnostic 协议的长任务流实验，我们将其列为未来工作而非在本文主张。

## W3 规模/信噪比不足，且隐藏两个空结果

已修复。新增 bootstrap 95% 置信区间（`experiments/paper_suite/bootstrap_ci.json`，脚本 `experiments/bootstrap_ci.py`）：表1 关键差异区间 [−0.1042,+0.0208]、表3 区间触零，全部如实写入附录 `sec:bootstrap`。两个空结果在正文显式讨论：`no_transport` 与 `sinkhorn_topk` 逐位相同（0.4844/−0.0938）→ 传输项在该规模无可测效果，应通过 transport-gap 轨迹评估；四种自适应策略聚合统计相同 → 控制器差异不可测，表述改为"同源变量可导出、可监测"。EWC/SI/MAS 的 λ 网格（{100,1000,5000} / {0.1,1,10}）报告于表3 与 `r3_baseline_comparison.json`，且网格在该规模平坦。

## W4 基线对比存在 replay 混淆

已修复。表3 重构为统一协议：所有方法行（fine_tune / EWC / SI / MAS / Continual ASAM task / prototype）均以 `replay_batch_size=0` 运行，隔离 CL 机制本身；新增 ER 与 A-GEM 作为 replay 参照行（replay batch size 4）。config 快照（`r3_baseline_comparison.json` 中 `method_rows_replay_batch_size=0`、`er_agem_replay_batch_size=4`）随产物发布。新结果：prototype 取得最高平均准确率 0.5156±0.0156（遗忘 0.0312±0.0312），ER/A-GEM 为 0.5000/0.0000，fine_tune 为 0.4844/0.0000；差异均在 seed 噪声内（bootstrap 区间触零）。

## W5 理论贡献为记账式结果

承认。理论章节重定位为"监控与诊断变量"：定理1 是 transport loss 的 ε-一致性陈述（含有效上界 OT(C)+ε·KL(π*‖rcᵀ)），定理2 是路由 KL 的定义分解，定理3 是 excess 的几何代理；不主张新界、新收敛率或可证伪预测。摘要、贡献列表、结论三处措辞已统一。

## 可验证性

- 守卫测试：`python -m pytest tests/test_paper_artifacts_consistency.py -q`（表1/2/3、摘要、单跑诊断与产物逐字段一致，旧数字反断言）。
- 复现表1：`python experiments/run_continual_text_ablation.py --protocol task_incremental_multihead --dataset-name split_ag_news --vocab-size 10000 --dim 128 --num-heads 8 --num-layers 2 --epochs-per-task 3 --num-seeds 3 --max-train-samples 64 --max-val-samples 32 --device cpu --output-json experiments/paper_suite/r2_agnews_bpe_3ep.json`
- 复现表3：`python experiments/run_baseline_comparison.py --num-seeds 2 --epochs-per-task 1 --dim 64`（输出 `experiments/paper_suite/r3_baseline_comparison.json`）
- 复现 bootstrap：`python experiments/bootstrap_ci.py`（输出 `experiments/paper_suite/bootstrap_ci.json`）
- 表2 权威产物：`experiments/paper_suite/continual_operator_ablation.json`（由 `experiments/run_continual_operator_ablation.py` 生成，config 内嵌）。

## 提交索引

- `b1c61b1` 论文：摘要归位、空结果讨论、贡献重定位
- `f002dee` 实验：统一 replay 表3 + ER/A-GEM + λ 网格 + bootstrap
- `9943e90` 测试：摘要/表3 守卫 + 基线辅助函数测试
- `b529eef` 文档：计划收尾
