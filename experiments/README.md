# GTX 3060 12GB 实验指南

## 快速开始

### 1. 安装依赖

```bash
# 确保安装了必要的包
pip install torch matplotlib numpy

# 如果还没安装 ASAM
pip install -e ..
```

### 2. 运行基准测试

```bash
# 进入实验目录
cd experiments

# 运行 3060 优化版实验
python run_3060_baseline.py
```

### 3. 预期输出

实验会逐步测试以下内容：

#### Test 1: Forward Pass Speed
- 序列长度: 128, 256, 512, 1024, 1536, 2048
- 对比: ASAM vs Standard Transformer
- 预期: ASAM 应该更快，且能处理更长的序列

#### Test 2: Training Speed
- 50 步训练
- 监测损失下降和速度
- 预期: 验证梯度是否正常流动

#### Test 3: Sparse Patterns
- 对比 Local / Strided / Hierarchical 模式
- 预期: Hierarchical 综合性能最好

#### Test 4: Adaptive Gate
- 测试不同输入类型下的门控行为
- 预期: 简单输入使用更多稀疏注意力

## 实验参数说明

### 保守设置（不会 OOM）

```python
# 模型配置
dim=256              # 从 512 减小
num_heads=4          # 从 8 减小
num_layers=2         # 从 4-6 减小
batch_size=2         # 保持较小
max_seq_len=2048     # 超过会 OOM
```

### 显存使用估算

| 序列长度 | 显存占用 | 状态 |
|---------|---------|------|
| 128-512 | ~2-4 GB | ✅ 安全 |
| 1024 | ~6-8 GB | ✅ 可以 |
| 1536 | ~9-10 GB | ⚠️ 接近极限 |
| 2048 | ~11 GB | ⚠️ 刚好能用 |
| >2048 | >12 GB | ❌ OOM |

## 运行时间预估

- **Total Runtime**: 30-60 分钟（取决于 GPU）
- **Test 1**: 10-15 分钟
- **Test 2**: 5-10 分钟
- **Test 3**: 5-10 分钟
- **Test 4**: 2-5 分钟

## 结果查看

实验完成后，结果保存在：

```
experiments/results_3060/
├── results_YYYYMMDD_HHMMSS.json   # 详细数据
└── plots_YYYYMMDD_HHMMSS.png      # 可视化图表
```

### 关键指标

1. **速度提升**: ASAM vs Standard 的时间比
2. **内存节省**: Peak Memory 对比
3. **稀疏比例**: Hierarchical 模式的稀疏率
4. **门控行为**: 不同输入下的 gate value 范围

## 如果出现问题

### OOM (Out of Memory)

如果看到 OOM 错误：
1. 脚本会自动捕获并清理显存
2. 实验会继续下一个测试
3. 结果是正常的，说明到了显存极限

### 运行太慢

可以调整参数加速：
```python
# 在 run_3060_baseline.py 中修改
num_steps=20  # 从 50 减小
seq_lengths = [128, 256, 512, 1024]  # 去掉长的
```

## 下一步

### 如果 3060 实验成功 ✅

1. 分析结果，确认 ASAM 在小规模有效
2. 准备云端实验（需要更大显存做长序列）
3. 收集真实数据，更新论文

### 资源推荐

| 平台 | 适合场景 | 费用 |
|------|---------|------|
| Google Colab (免费) | 单次长序列实验 | 免费 |
| Kaggle (免费) | 批量实验 | 免费 (30h/周) |
| AutoDL (国内) | 长期租用 | ~¥1-2/小时 |

## 联系

如果实验遇到问题，可以：
1. 查看 `results_*.json` 中的详细日志
2. 检查 GPU 状态: `nvidia-smi`
3. 清理显存: `python -c "import torch; torch.cuda.empty_cache()"`
