# 创建 Classic PAT（推荐）

## 步骤

1. 访问 https://github.com/settings/tokens

2. 点击 **"Generate new token"** (下拉菜单) → **"Generate new token (classic)"**

3. 填写信息：
   - **Note**: `ASAM Push Token`
   - **Expiration**: 选择 30 days 或 No expiration
   - **Select scopes**: 勾选 **repo**（完整控制仓库）

4. 点击 **Generate token**

5. **立即复制 Token**（只显示一次！）

格式：`ghp_xxxxxxxxxxxxxxxxxxxx`

6. 更新到 `.env` 文件：
   ```
   GITHUB_TOKEN=ghp_xxxxxxxx
   ```

7. 再次推送：
   ```powershell
   python push_to_github.py
   ```
