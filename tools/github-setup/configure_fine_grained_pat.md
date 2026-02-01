# Fine-grained PAT 正确配置方法

## 步骤 1：创建 Token

1. 访问 https://github.com/settings/tokens?type=beta

2. 点击 **"Generate new token"**

3. 填写信息：
   - **Token name**: `ASAM Push Token`
   - **Expiration**: 30 days
   - **Description**: Token for pushing ASAM code

## 步骤 2：选择仓库访问权限

在 **"Repository access"** 部分选择：

- ✅ **All repositories**（推荐，简单）

或

- ✅ **Only select repositories** → 选择 `li-guohao/asam-attention`

## 步骤 3：设置权限（关键！）

展开 **"Repository permissions"**，找到：

- **Contents**: 选择 **Read and write**
  - 用于读取和写入仓库内容
  
- **Metadata**: 选择 **Read**
  - 用于搜索仓库

其他权限可以都不勾选。

## 步骤 4：生成 Token

点击 **"Generate token"**

**立即复制 Token**（格式：`github_pat_xxxxxxxx`，只显示一次！）

## 步骤 5：更新配置

编辑 `.env` 文件：
```
GITHUB_TOKEN=github_pat_你的token
```

## 步骤 6：测试推送

```powershell
python push_to_github.py
```

---

## ⚠️ 常见错误

如果还是 401 错误，检查：

1. **Token 是否真的复制完整了？**
   - Fine-grained PAT 很长，确保没有遗漏字符

2. **权限是否正确？**
   - Contents 必须是 **Read and write**，不是 Read only

3. **仓库访问范围是否正确？**
   - 确保包含了 `li-guohao/asam-attention`

4. **Token 是否过期？**
   - 检查 Expiration 日期
