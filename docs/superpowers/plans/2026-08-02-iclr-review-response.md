# ICLR 审稿意见应对计划（第二阶段整改）
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
**Goal:** 针对 ICLR 模拟审稿（3/10，5 条 Critical）在修订期内消除可修复缺陷，并把论文重定位为可辩护的"持续稀疏路由诊断框架"；长任务流实验作为可选项。
**Architecture:** 继续以 repo_tmp 为项目根（master=1d48fbf，GitHub 已同步）；所有实验改动经守卫测试与 pdflatex 编译验证后分块提交、推送。可修复项（摘要/公平性/空结果披露）先行，结构性重定位靠重写解决，长实验仅按需执行。
**Tech Stack:** Python 3.12 + PyTorch 2.11（venv），TinyTeX pdflatex，pytest 守卫测试，HuggingFace offline 缓存（ag_news/cifar10）。
---
## 决策点（执行前确认；默认值见标注）
- D1 投稿目标：A) ICLR 主刊（必须含 Phase 3 长实验）  B) workshop/rebuttal（Phase 0–2 足够）【默认 B】
- D2 replay 公平性：A) 全部方法无 replay（需重跑表1/表3）  B) 全部统一 replay 并新增 ER/A-GEM 基线（默认 B，改动可控）
- D3 摘要数字：A) 删除 +0.0625/0.4844→0.5156 的拼接表述  B) 保留并把 continual_ablation.json 列入权威产物清单、标注来源【默认 A】
## Phase 0: 摘要数字归位与一致性（最高优先，可修订期）
**Files:** paper/continual_asam.tex, tests/test_paper_artifacts_consistency.py
- [x] 按 D3 处理摘要中来自两套不同实验的拼接数字（+0.0625→−0.0938 出自 continual_ablation.json；0.4844→0.5156 出自 operator ablation 的 sinkhorn→kl_topk）
- [x] 摘要末句删除"connect routing geometry diagnostics to forgetting"，改为"导出阶段轨迹供监测"（与正文第207/365行一致）
- [x] Reproducibility 段落权威产物清单与摘要引用一致（若选 D3-B 则加入 continual_ablation.json）
- [x] 更新守卫测试：把摘要关键数字纳入断言或显式豁免，防止再次漂移
## Phase 1: 实验公平性与空结果披露（可修订期）
**Files:** experiments/run_baseline_comparison.py, experiments/paper_suite/r2_baseline_comparison.json, paper/continual_asam.tex
- [x] 按 D2 统一 replay：新增 ER/A-GEM 基线（复用 TaskHeadTransformer + ReplayBuffer），ASAM 行与基线行的 replay 设置明确一致；JSON config 记录 replay 对齐方式
- [x] 报告 EWC/SI/MAS 的 λ 及调优网格（写入 JSON config 与论文附录）
- [x] 显式讨论空结果 1：no_transport 与 sinkhorn_topk 逐位相同（0.4844/−0.0938）→ 传输项在该规模无可测效果
- [x] 显式讨论空结果 2：四种自适应策略聚合统计相同 → 控制器差异在该规模不可测，改为"同源变量可导出、可监测"表述
- [x] bootstrap 置信区间：表1（0.0000 vs 0.0417）、表3（0.1250），脚本输出到实验产物
## Phase 2: 论文重定位（结构性，靠重写解决）
**Files:** paper/continual_asam.tex
- [x] 主贡献改为：(i) 同源变量闭环设计（同一变量用于优化/生命周期/诊断）；(ii) 可追溯、有守卫测试的评测 pipeline；(iii) 诊断量有效性边界的诚实刻画
- [x] "降低遗忘"降级为初步观察；删除"multi-seed ablation across routing strategies"作为贡献的表述
- [x] 理论章节定位为"监控与诊断变量"；摘要/贡献列表/结论三处措辞一致性收尾
## Phase 3: 长任务流实验（结构性，完整实验周期；仅 D1-A 需要）
**Files:** experiments/（新增 runner），experiments/paper_suite/r3_*（新产物）
- [ ] Split AG News + Split CIFAR-10，T=5–10，class-incremental single-head（无 oracle id），每任务 64–256 样本
- [ ] 展示诊断量（transport gap/excess/routing stability）在真实遗忘体制下随阶段的移动
- [ ] 验证 no_transport 在更大规模下是否产生可测差异（回应"诊断无意义"攻击）
- [ ] r3_* 产物接入守卫测试与 Reproducibility 清单
## Phase 4: 收尾与发布
- [x] pdflatex 重编译（TinyTeX，16 页目标，无 undefined）
- [x] 守卫 + 回归测试全绿（tests/test_paper_artifacts_consistency.py 等，70 passed）
- [x] 逻辑分块提交（摘要/基线/重定位/长实验）并推送 GitHub（直连代理覆盖）
- [x] 若走 rebuttal：整理攻击面回应要点（承认定位、ER/λ/T≥5 证据、可复现性作为贡献）
## 验收
- 摘要每个数字可回溯到列出的权威产物；表3 replay 设置公开且 ER 基线齐全；两个空结果在正文显式讨论；守卫测试全绿；PDF 编译通过；GitHub 同步（master=本地 HEAD）。
