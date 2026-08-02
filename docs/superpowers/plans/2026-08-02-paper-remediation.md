# ASAM 论文整改与可复现性修复 Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
**Goal:** 消除论文中不可回溯/自相矛盾的实证数字，重跑权威实验，修正理论过度声称，并修复静默数据回退等工程缺陷，使论文主张与仓库产物一一对应。
**Architecture:** 以 `repo_tmp` 为项目根（嵌套 git 仓库，当前分支 master）；用 venv `C:\Users\Administrator\.codex\worktrees\b51b\2GPU\.venv\Scripts\python.exe`（torch 2.11.0+cu128）执行重跑，结果沿用 `device=cpu` 与现有产物可比；HF 数据走离线缓存（`HF_DATASETS_OFFLINE=1`）。论文修改落在 `repo_tmp/paper/continual_asam.tex`。
**Tech Stack:** Python 3.12.13, PyTorch 2.11, HuggingFace datasets (offline cache), matplotlib, LaTeX (可选), pytest。
---
## Task 1: 环境准备
**Files:**
- Env: venv at `C:\Users\Administrator\.codex\worktrees\b51b\2GPU\.venv`
- [ ] **Step 1: 安装缺失依赖**
  - 运行: `C:\Users\Administrator\.codex\worktrees\b51b\2GPU\.venv\Scripts\python.exe -m pip install datasets matplotlib`
  - 预期: 安装成功；随后 `pip check` 无冲突。
- [ ] **Step 2: 验证离线数据可用**
  - 设 `HF_DATASETS_OFFLINE=1` 后运行: `python -c "from datasets import load_dataset; d=load_dataset('ag_news', split='train'); print(len(d))"`
  - 预期: 从缓存加载成功（ag_news 已缓存于 `~/.cache/huggingface/datasets/ag_news`）。
- [ ] **Step 3: 验证 `datasets.text_dataset.AGNewsDataset.load` 走真实数据**
  - 在 `repo_tmp` 下运行小脚本，检查返回数据集非 dummy（`data_source == 'huggingface'` 或文本含真实标题）。
  - 若安装或缓存加载失败：停止并报告，不进入重跑。
## Task 2: 重跑表1（BPE 策略消融，3 epochs）
**Files:**
- Run: `experiments/run_continual_text_ablation.py`
- Output: `experiments/paper_suite/r2_agnews_bpe_3ep.json` + 每 seed/策略 JSON + table/csv/png
- [ ] **Step 1: 核对 CLI 参数**
  - 运行: `python experiments/run_continual_text_ablation.py --help`
  - 确认 `--epochs-per-task`、`--num-seeds`、`--vocab-size`、`--dim`、`--num-heads`、`--num-layers`、`--max-train-samples`、`--max-val-samples`、`--device`、`--output-json` 存在。
- [ ] **Step 2: 运行消融（5 策略 × 3 seeds，CPU）**
  - 在 `repo_tmp` 下、设 `HF_DATASETS_OFFLINE=1`：
    `python experiments/run_continual_text_ablation.py --protocol task_incremental_multihead --dataset-name split_ag_news --vocab-size 10000 --dim 128 --num-heads 8 --num-layers 2 --epochs-per-task 3 --num-seeds 3 --max-train-samples 64 --max-val-samples 32 --device cpu --output-json experiments/paper_suite/r2_agnews_bpe_3ep.json`
  - 预期: 15 runs 完成（预计 30–60 分钟），生成聚合与 per-seed 产物。
- [ ] **Step 3: 校验产物**
  - 解析 `r2_agnews_bpe_3ep.json`：`num_runs==3`；config `epochs_per_task==3`、`vocab_size==10000`、`dim==128`、`num_layers==2`、`max_train_samples==64`、`max_val_samples==32`；记录聚合表数字。
## Task 3: 重跑表3（基线对比，统一协议）
**Files:**
- Modify: `experiments/run_baseline_comparison.py`
- Output: `experiments/paper_suite/r2_baseline_comparison.json`
- [ ] **Step 1: 改造脚本**
  - 与主消融共享数据管线：`datasets.text_dataset.get_continual_dataloaders`，char tokenizer，`max_train_samples=64`、`max_val_samples=32`、`classes_per_task=2`、task-incremental multi-head、oracle task id。
  - 六行统一骨干规模（D=64、1 层、4 头、每任务分类头）；仅 CL 机制不同（fine_tune / EWC / SI / MAS / Continual ASAM task_routing / prototype）。
  - EWC/SI/MAS 经 `asam.continual_baselines` 复用，加薄适配器支持 `forward(x, task_ids)`。
  - 输出 JSON 内嵌完整 config。
- [ ] **Step 2: 运行**
  - `python experiments/run_baseline_comparison.py --num-seeds 2 --epochs-per-task 1 --dim 64 --output-json experiments/paper_suite/r2_baseline_comparison.json`
  - 预期: 6 方法 × 2 seeds（预计 15–30 分钟）。
- [ ] **Step 3: 校验**
  - JSON 含 config 快照与 6 行聚合（accuracy/forgetting/BWT），记录数字。
## Task 4: 论文修订（continual_asam.tex）
**Files:**
- Modify: `paper/continual_asam.tex`
- Create: `paper/references.bib`
- [ ] **Step 1: 替换表1/表3 数字**
  - 用 Task 2/Task 3 新产物聚合值替换；表注改为真实 seeds（表1: 3 seeds；表3: 2 seeds）。
- [ ] **Step 2: 对齐摘要/设置/路由描述**
  - 删除 0.5521 与 0.6250 矛盾表述；删除"256 训练/128 验证"；"M=T"改为"T×prototype_slots_per_task（默认 2 槽/任务）"；实验设置段写真实 caps/seeds/epochs。
- [ ] **Step 3: 理论修正并降级**
  - 定理1: ε→0 收敛到精确 OT；有效上界 ⟨T*,C⟩ ≤ OT(C)+ε·KL(π*‖rcᵀ)；删除 "≤ε·KL+O(ε²)" 链条与 κ 不等式。
  - 定理2: 保留恒等式，删除"熵升=探索/交叉熵降=覆盖"归因。
  - 定理3: 标注"碰撞是代理指标而非遗忘上界"。
  - 推论1: 删除定理环境，改为明确标注的非形式化讨论段落。
  - 贡献第 4 条与摘要弱化。
- [ ] **Step 4: 诊断章节**
  - "与遗忘相关"→"导出阶段轨迹；相关分析需更长任务流（未来工作）"。
- [ ] **Step 5: 新建 references.bib**
  - 收录论文引用的 19 个文献条目。
- [ ] **Step 6: TeX 校验**
  - 有工具链则用 latex:latex-compile 编译；否则做括号/环境配平检查。
## Task 5: 代码健壮性
**Files:**
- Modify: `datasets/text_dataset.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- [ ] **Step 1: 回退门控**
  - 默认加载失败时 raise（带醒目警告）；显式设 `ASAM_ALLOW_DATASET_FALLBACK=1` 才允许 dummy；数据集对象记录 `data_source`。
- [ ] **Step 2: README/CHANGELOG**
  - README v1.1.1 → v1.2.0；CHANGELOG 加一条修复记录。
- [ ] **Step 3: 不改动 replay buffer 与模型行为**（仅确认不改）。
## Task 6: 一致性守卫测试
**Files:**
- Create: `tests/test_paper_artifacts_consistency.py`
- Modify: `tests/`（回退行为测试）
- [ ] **Step 1: 新增一致性测试**
  - 从 tex 正则提取表1/表2/表3/摘要关键数字，与 `r2_agnews_bpe_3ep.json`、`continual_operator_ablation.json`、`r2_baseline_comparison.json`、`continual_benchmark.json` 逐项断言相等。
- [ ] **Step 2: 回退行为测试**
  - 默认 raise；env 放行；离线缓存加载真实数据。
- [ ] **Step 3: 回归**
  - 离线跑 `tests/test_continual_real_benchmark.py`、`tests/test_paper_pipeline.py`、`tests/test_datasets_and_visualization.py`。
## Task 7: 入库与记录
**Files:**
- Commit in `repo_tmp`
- Modify: `E:\ASAM Adaptive Sparse Attention Module\findings.md`、`task_plan.md`、`progress.md`
- [ ] **Step 1: 逻辑分块提交**（仅暂存本次涉及路径）：
  1. 实验脚本 + 新产物
  2. 论文 tex + references.bib
  3. 回退修复 + 测试
  4. README/CHANGELOG
- [ ] **Step 2: 更新根目录记录文件**
## 验收
- tex 无 0.5521/0.6250 矛盾；所有表格数字可指到仓库内产物；理论章节无错误证明链；`references.bib` 存在；README 版本一致；指定测试通过。
