# GitHub 配置与推送指南

这个指南帮助你安全地将代码推送到 GitHub，**不会泄露你的 Token**。

---

## 📋 文件说明

| 文件 | 用途 |
|------|------|
| `check_setup.py` | 检查配置是否正确 |
| `setup_github_token.bat` | 设置 Token (Windows CMD) |
| `setup_github_token.ps1` | 设置 Token (PowerShell) |
| `push_to_github.py` | 推送代码到 GitHub |
| `.env.example` | 环境变量模板 |

---

## 🚀 快速开始（3步）

### 第 1 步：检查当前配置

```powershell
python check_setup.py
```

这个命令会告诉你：
- ✅ Token 是否已设置
- ✅ Git 仓库是否正常
- ✅ 能否连接到 GitHub

**如果显示 "所有检查通过"**，直接跳到第 3 步！

---

### 第 2 步：设置 GitHub Token

#### 方法 A：使用批处理脚本（最简单）

```powershell
# 在 CMD 或 PowerShell 中运行
.\setup_github_token.bat
```

然后按提示输入你的 Token。

#### 方法 B：使用 PowerShell 脚本

```powershell
# 在 PowerShell 中运行
.\setup_github_token.ps1
```

输入时 Token 会被隐藏（安全）。

#### 方法 C：手动设置

```powershell
# 临时设置（仅当前窗口有效）
$env:GITHUB_TOKEN = "ghp_xxxxxxxx"

# 永久设置（推荐）
[Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "ghp_xxxxxxxx", "User")
```

**获取 Token**: https://github.com/settings/tokens

需要的权限：
- ✅ `repo` (访问仓库)

---

### 第 3 步：推送代码

```powershell
python push_to_github.py
```

这个脚本会：
1. 读取环境变量中的 Token（安全）
2. 配置 Git 远程仓库
3. 推送代码到 GitHub

---

## 🔍 常见问题

### Q1: 如何验证 Token 是否设置成功？

```powershell
# 查看环境变量
echo $env:GITHUB_TOKEN

# 应该显示你的 Token（开头部分）
# 如: ghp_xxxxxxxx...
```

### Q2: 推送时提示 "Authentication failed"

**原因**: Token 无效或过期

**解决**:
1. 访问 https://github.com/settings/tokens
2. 检查 Token 是否过期
3. 生成新的 Token
4. 重新运行 `setup_github_token.bat`

### Q3: 推送时提示 "rejected"

**原因**: 远程仓库有更新，本地不是最新

**解决**:
```bash
git pull origin main --rebase
git push origin main
```

### Q4: 我不想用环境变量，还有其他方法吗？

**方法 A：Git Credential Manager**

```bash
git push origin main
# 第一次输入用户名和密码后选择"记住"
```

**方法 B：SSH 密钥**（最安全）

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your@email.com"

# 添加到 GitHub
# https://github.com/settings/keys

# 使用 SSH 地址
git remote set-url origin git@github.com:li-guohao/asam-attention.git
```

---

## 🛡️ 安全提示

### ✅ 正确的做法

- ✅ 使用环境变量存储 Token
- ✅ 定期更换 Token（每 3-6 个月）
- ✅ Token 只给最小必要权限
- ✅ 启用 GitHub 两步验证

### ❌ 错误的做法

- ❌ 将 Token 写入代码文件
- ❌ 将 Token 提交到 Git
- ❌ 在聊天记录中发送 Token
- ❌ 分享 Token 给他人

---

## 📁 完整工作流程示例

```powershell
# 1. 进入项目目录
cd e:\GIT\asam-attention

# 2. 检查配置
python check_setup.py

# 3. 如果有问题，设置 Token
.\setup_github_token.bat

# 4. 再次检查
python check_setup.py

# 5. 提交代码（如果有更改）
git add .
git commit -m "你的提交信息"

# 6. 推送到 GitHub
python push_to_github.py

# 7. 打开 GitHub 查看
start https://github.com/li-guohao/asam-attention
```

---

## 🆘 遇到问题？

1. **查看错误信息**: 脚本会给出具体的错误原因
2. **运行检查工具**: `python check_setup.py`
3. **检查网络**: 确保能访问 https://github.com
4. **重新设置 Token**: 运行 `setup_github_token.bat`

---

**安全提醒**: 如果你不小心泄露了 Token，请立即到 https://github.com/settings/tokens 删除并重新生成！
