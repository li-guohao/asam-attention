# GitHub 安全最佳实践

## 🎯 目标
防止敏感信息（Token、密码、API Key）泄露到 GitHub

---

## 📁 文件分类

### 🔴 绝对不能提交的文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `.env` | 真实 Token | ❌ 已在 .gitignore |
| `.env.local` | 本地配置 | ❌ 已在 .gitignore |
| `*.secret` | 密钥文件 | ❌ 已在 .gitignore |
| `.github_config.ini` | GitHub配置 | ❌ 已在 .gitignore |

### ✅ 可以提交的文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `.env.example` | 配置模板（无真实值） | ✅ 可以提交 |
| `README.md` | 项目说明 | ✅ 可以提交 |
| `setup.py` | 项目配置 | ✅ 可以提交 |

---

## 🔍 提交前检查清单

每次 `git commit` 前执行：

```powershell
# 1. 查看哪些文件会被提交
git status

# 2. 确认没有看到 .env（带真实token的）
# 应该只看到 .env.example

# 3. 查看具体改动
git diff --cached

# 4. 确认安全后再提交
git commit -m "提交信息"
```

---

## 🚨 如果不小心提交了敏感信息

### 步骤 1：立即撤销 Token
```
访问 https://github.com/settings/tokens
→ 找到对应的 Token → Delete
```

### 步骤 2：生成新 Token
```
https://github.com/settings/tokens/new
勾选 repo 权限
```

### 步骤 3：清理 Git 历史
```bash
# 如果已经推送到 GitHub，需要强制重写历史
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch .env' \
--prune-empty --tag-name-filter cat -- --all

# 然后强制推送
git push origin main --force
```

⚠️ **注意**: 强制推送会影响团队协作，仅用于个人项目！

---

## 💡 最佳实践总结

### ✅ 应该做的

1. **使用 .env 文件存储敏感信息**
   ```
   GITHUB_TOKEN=ghp_xxxxxxxx
   ```

2. **提供 .env.example 模板**
   ```
   GITHUB_TOKEN=your_token_here
   ```

3. **每次提交前检查 `git status`**

4. **使用 pre-commit 钩子**（已配置）

5. **定期更换 Token**（每3-6个月）

6. **给 Token 最小权限**
   - 只勾选必要的权限（如 repo）
   - 设置过期时间

### ❌ 不应该做的

1. ❌ 不要将真实 Token 写入代码文件
2. ❌ 不要在聊天记录中发送 Token
3. ❌ 不要截图包含 Token 的屏幕
4. ❌ 不要分享 Token 给他人
5. ❌ 不要使用过期的 Token

---

## 🔧 工具推荐

### 1. git-secrets（自动检测敏感信息）
```bash
# 安装
git clone https://github.com/awslabs/git-secrets
cd git-secrets && make install

# 配置
git secrets --install
git secrets --register-aws
git secrets --add 'github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}'
```

### 2. GitGuardian（监控泄露）
```
https://www.gitguardian.com/
自动扫描 GitHub 仓库中的敏感信息
```

### 3. GitHub Secret Scanning
```
GitHub 自动扫描推送的代码中的已知密钥格式
```

---

## 📞 紧急联系

如果发现敏感信息泄露：

1. **立即撤销 Token**（最重要！）
   https://github.com/settings/tokens

2. **检查仓库访问日志**
   https://github.com/li-guohao/asam-attention/security

3. **联系 GitHub 支持**
   https://support.github.com/

---

## ✅ 当前项目安全检查

| 检查项 | 状态 |
|--------|------|
| .env 在 .gitignore | ✅ |
| .env.example 无真实值 | ✅ |
| pre-commit 钩子配置 | ✅ |
| 敏感文件未跟踪 | ✅ |

**结论：当前配置安全，可以提交！**
