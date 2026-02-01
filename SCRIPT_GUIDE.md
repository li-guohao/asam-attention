# GitHub 辅助脚本使用指南

## 📦 已创建的文件

### 🔧 核心脚本

| 文件名 | 大小 | 用途 | 使用场景 |
|--------|------|------|----------|
| `check_setup.py` | ~10KB | 配置检查工具 | 验证 Token 和 Git 配置 |
| `push_to_github.py` | ~8KB | 推送助手 | 安全推送代码到 GitHub |
| `setup_github_token.bat` | ~1KB | Token 设置 (CMD) | Windows 命令行用户 |
| `setup_github_token.ps1` | ~3KB | Token 设置 (PowerShell) | PowerShell 用户 |

### 📄 辅助文件

| 文件名 | 大小 | 用途 |
|--------|------|------|
| `.env.example` | ~0.6KB | 环境变量模板 |
| `GITHUB_SETUP_GUIDE.md` | ~4KB | 详细使用说明 |
| `SECURITY_CHECK_REPORT.md` | ~4KB | 安全检查报告 |

---

## 🎯 使用流程

### 第一次使用

```
1. check_setup.py       ← 检查当前配置
         ↓
2. setup_github_token.bat  ← 设置 Token（如果缺少）
         ↓
3. check_setup.py       ← 再次验证
         ↓
4. push_to_github.py    ← 推送代码
```

### 日常使用

```
直接运行: push_to_github.py
```

---

## 🔐 安全特性

所有脚本都遵循以下安全原则：

1. **Token 不落地**
   - Token 只存在于环境变量
   - 不会写入任何文件
   - 屏幕输出时会隐藏

2. **Git 历史保护**
   - Token 不会被提交
   - `.gitignore` 已配置

3. **错误处理**
   - 详细的错误提示
   - 常见的解决方案
   - 不会暴露敏感信息

---

## 📝 脚本详细说明

### 1. check_setup.py

**功能**: 全面检查配置状态

**检查项**:
- 环境变量 GITHUB_TOKEN
- Git 仓库配置
- 远程仓库设置
- GitHub 连接测试

**输出示例**:
```
========================================
步骤 1: 检查 GitHub Token
========================================
✅ 找到 GITHUB_TOKEN 环境变量
   Token 前缀: ghp_xxxxx...
   Token 长度: 40 字符
✅ Token 格式正确 (Classic)

========================================
步骤 2: 检查 Git 仓库
========================================
✅ 当前目录是 Git 仓库
✅ 已配置远程仓库 (origin)
✅ 当前分支: main
⚠️  有未提交的更改

========================================
步骤 3: 测试 GitHub 连接
========================================
✅ 成功连接到 GitHub！
   用户名: li-guohao
```

### 2. push_to_github.py

**功能**: 安全推送代码

**特点**:
- 自动读取环境变量
- 配置远程仓库
- 详细的推送状态
- 错误诊断

**使用**:
```powershell
python push_to_github.py
```

### 3. setup_github_token.bat

**功能**: 设置环境变量（CMD 版本）

**特点**:
- 简单易用
- 永久生效
- 输入可见（注意周围无人）

**使用**:
```powershell
.\setup_github_token.bat
```

### 4. setup_github_token.ps1

**功能**: 设置环境变量（PowerShell 版本）

**特点**:
- 输入时隐藏 Token
- 更安全的输入方式
- 验证 Token 格式

**使用**:
```powershell
.\setup_github_token.ps1
```

---

## 🆘 故障排除

### 问题 1: "找不到 GITHUB_TOKEN"

**解决**:
```powershell
# 临时设置
$env:GITHUB_TOKEN = "你的token"

# 或者运行设置脚本
.\setup_github_token.bat
```

### 问题 2: "Token 无效"

**解决**:
1. 访问 https://github.com/settings/tokens
2. 确认 Token 未过期
3. 确认勾选了 `repo` 权限
4. 重新生成并设置

### 问题 3: "无法连接到 GitHub"

**可能原因**:
- 网络问题
- 代理设置
- 防火墙

**测试**:
```powershell
# 测试网络
ping github.com

# 测试 HTTPS
curl https://api.github.com
```

---

## 📚 推荐阅读顺序

1. **首先阅读**: `GITHUB_SETUP_GUIDE.md`
   - 完整的使用流程
   - 常见问题解答

2. **然后运行**: `check_setup.py`
   - 了解当前状态
   - 发现配置问题

3. **最后使用**: `push_to_github.py`
   - 实际推送代码

---

## 🎨 自定义配置

如果你需要修改默认配置，编辑脚本中的这些变量：

### push_to_github.py

```python
# 第 85 行附近
def get_git_info():
    github_username = "li-guohao"  # 修改为你的用户名
    repo_name = "asam-attention"   # 修改为你的仓库名
```

### check_setup.py

```python
# 第 180 行附近
if not token.startswith(("ghp_", "github_pat_")):
    # 如果你使用其他格式的 Token，修改这里
```

---

## ✅ 检查清单

发布前确认：

- [ ] 运行 `check_setup.py` 无错误
- [ ] Token 已设置且未过期
- [ ] 能连接到 GitHub
- [ ] Git 仓库已初始化
- [ ] 代码已提交
- [ ] 远程仓库 URL 正确

---

**注意**: 所有脚本都已添加详细注释，你可以直接阅读源码了解实现细节。
