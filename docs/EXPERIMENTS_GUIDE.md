# 在 GTX 3060 12GB 上运行 ASAM 实验

## ⚠️ 重要说明

**我无法直接在你的GPU上运行实验**，因为我运行在一个远程服务器环境中。

但我为你准备了**完整的可执行脚本**，你只需要在你的机器上运行即可。

---

## 🚀 快速开始（3个简单步骤）

### 步骤 1：下载代码

```bash
git clone https://github.com/li-guohao/asam-attention.git
cd asam-attention
```

### 步骤 2：安装依赖

**Windows用户**：
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy seaborn
```

**Linux用户**：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy seaborn
```

### 步骤 3：运行实验

```bash
cd experiments
python run_3060_baseline.py
```

---

## 📋 实验内容

运行后会自动执行以下测试：

| 测试项 | 目的 | 预计时间 |
|--------|------|---------|
| Forward Speed | 测试推理速度 | 10-15 min |
| Training Speed | 验证梯度流动 | 5-10 min |
| Sparse Patterns | 对比不同稀疏模式 | 5-10 min |
| Adaptive Gate | 验证自适应门控 | 2-5 min |

**总时间**: 约 30-60 分钟

---

## 📊 预期结果

### ✅ 成功标志

1. **速度提升**: ASAM 比 Standard Transformer 快 **1.5-4倍**
2. **内存节省**: Peak memory 减少 **2-4倍**
3. **门控工作**: 简单输入 sparse ratio > 60%
4. **无NaN**: 梯度正常，损失下降

### ⚠️ OOM是正常的

当序列长度 > 2048 时，Standard Transformer 会OOM，这是预期的：
```
Seq Len 2048: Standard -> OOM (expected)
Seq Len 2048: ASAM     -> 11GB (works!)
```

---

## 🔧 如果出错

### 问题1: "No module named 'torch'"

**解决**：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 问题2: "CUDA out of memory"

**解决**：这是正常的，脚本会自动处理。如果频繁OOM，修改参数：

```python
# 在 run_3060_baseline.py 中减小这些值
seq_lengths = [128, 256, 512, 1024]  # 去掉 1536, 2048
batch_size = 1  # 从 2 改为 1
```

### 问题3: 运行太慢

**解决**：减小测试步数：
```python
num_steps = 20  # 从 50 减小
```

---

## 📁 结果查看

实验完成后，结果保存在：

```
experiments/results_3060/
?   ??? results_20260201_123045.json   # ?????????
?   ??? plots_20260201_123045.png      # ??????????
```

> Note: files under `experiments/results_3060/` are generated locally and are not tracked by Git.

### 关键指标解读

```json
{
  "forward_speed": {
    "ASAM": [
      {"seq_len": 512, "time_ms": 8.5, "memory_mb": 2200},
      {"seq_len": 1024, "time_ms": 18.2, "memory_mb": 4500}
    ]
  },
  "gate_behavior": [
    {"input_type": "Random", "gate_mean": 0.75, "sparse_ratio": 0.68}
  ]
}
```

**好的结果**：
- ASAM time < Standard time ✓
- Gate varies with input complexity ✓
- Sparse ratio > 50% ✓

---

## 🎯 下一步（云端扩展）

如果 3060 实验成功，可以用云端 GPU 跑更大规模：

### 免费选项

| 平台 | GPU | 时长 | 适合场景 |
|------|-----|------|---------|
| Google Colab | T4/V100 | 12h/天 | 长序列测试 |
| Kaggle | P100 | 30h/周 | 批量实验 |

### 脚本已准备

我已为你准备了云端脚本：
- `experiments/run_colab_lra.py` - Colab 完整 LRA
- `experiments/run_kaggle_benchmark.py` - Kaggle 对比实验

---

## 📞 需要帮助？

如果实验遇到问题：

1. **??????**: `experiments/results_3060/results_*.json`
2. **检查GPU**: `nvidia-smi`
3. **清理显存**: `python -c "import torch; torch.cuda.empty_cache()"`

---

## ✅ 检查清单

运行前确认：
- [ ] Python 3.8+ 已安装
- [ ] PyTorch with CUDA 已安装
- [ ] GTX 3060 驱动已安装
- [ ] 12GB 显存可用（关闭其他程序）
- [ ] 30-60分钟空闲时间

---

**现在就开始吧！运行 `python experiments/run_3060_baseline.py`**
