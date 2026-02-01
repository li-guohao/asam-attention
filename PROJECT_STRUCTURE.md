# ASAM 项目结构说明

```
asam-attention/
│
├── 📁 asam/                      # 核心库（主要代码）
│   ├── __init__.py
│   ├── asam_layer.py             # 原版 ASAM
│   ├── efficient_attention.py    # Flash Attention 优化
│   ├── asam_layer_optimized.py   # 真正稀疏注意力
│   ├── adaptive_gate.py
│   ├── sparse_patterns.py
│   └── ...
│
├── 📁 tests/                     # 单元测试
│   ├── test_basic.py
│   ├── test_efficient.py
│   └── test_asam.py
│
├── 📁 examples/                  # 使用示例
│   ├── basic_usage.py
│   └── optimized_usage.py
│
├── 📁 experiments/               # 实验和基准测试
│   ├── run_final_benchmark.py
│   ├── train_mixed_precision.py
│   └── results_3060/            # 实验结果
│
├── 📁 docs/                      # 文档
│   ├── README.md                 # API文档等
│   ├── analysis_report.md        # 性能分析报告
│   ├── security/                 # 安全相关文档
│   │   ├── SECURITY_BEST_PRACTICES.md
│   │   └── SECURITY_CHECK_REPORT.md
│   └── GITHUB_RELEASE_GUIDE.md   # GitHub发布指南
│
├── 📁 scripts/                   # 辅助脚本
│   └── analyze_paper.py          # 论文分析工具
│
├── 📁 tools/                     # 开发工具（非核心）
│   └── github-setup/             # GitHub配置辅助工具
│       ├── check_setup.py
│       ├── push_to_github.py
│       ├── diagnose_token.py
│       ├── setup_github_token.bat
│       ├── setup_github_token.ps1
│       ├── GITHUB_SETUP_GUIDE.md
│       └── ...
│
├── 📄 README.md                  # 项目主页（用户先看这个）
├── 📄 setup.py                   # Python包配置
├── 📄 requirements.txt           # 依赖列表
├── 📄 LICENSE                    # 许可证
├── 📄 .gitignore                 # Git忽略规则
├── 📄 .env.example               # 环境变量模板
│
└── 🔒 .env                       # 敏感配置（Git忽略）
    └── GITHUB_TOKEN=...          # 不要提交到Git！

```

---

## 🎯 核心文件 vs 辅助工具

### ✅ 核心项目文件（必须保留）

| 路径 | 说明 |
|------|------|
| `asam/` | ASAM注意力模块核心代码 |
| `tests/` | 单元测试 |
| `examples/` | 使用示例 |
| `experiments/` | 基准测试和实验 |
| `docs/` | 项目文档 |
| `README.md` | 项目介绍 |
| `setup.py` | 包配置 |
| `requirements.txt` | 依赖列表 |

### 🛠️ 辅助工具（可选，GitHub配置用）

| 路径 | 说明 |
|------|------|
| `tools/github-setup/` | GitHub配置和推送辅助脚本 |
| `scripts/` | 其他辅助脚本 |

---

## 🚀 用户路径

### 普通用户
1. 阅读 `README.md`
2. 安装：`pip install -e .`
3. 查看 `examples/` 学习使用
4. 运行 `tests/` 验证安装

### 开发者
1. 阅读 `docs/TECHNICAL.md`
2. 查看 `asam/` 源码
3. 修改代码
4. 运行 `tests/` 确保不破坏功能

### 发布者
1. 使用 `tools/github-setup/` 配置GitHub
2. 阅读 `docs/GITHUB_RELEASE_GUIDE.md`
3. 执行发布流程

---

## 🧹 清理建议

如果只需要核心功能，可以删除：
- `tools/` - GitHub配置工具（配置完成后不需要）
- `scripts/` - 辅助脚本
- `docs/security/` - 安全检查文档（发布前检查用）

**但必须保留**：
- `asam/` - 核心代码
- `tests/` - 测试
- `examples/` - 示例
- `README.md` - 项目说明
