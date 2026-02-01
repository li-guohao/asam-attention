#!/usr/bin/env python3
"""
Token 诊断工具
==============

检查 GitHub Token 是否有效
"""

import os
import sys
from pathlib import Path


def get_token():
    """从 .env 或环境变量获取 token"""
    # 环境变量
    token = os.getenv("GITHUB_TOKEN")
    
    # .env 文件
    if not token and Path(".env").exists():
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"\'')
                    break
    
    return token


def check_token():
    """检查 token 有效性"""
    token = get_token()
    
    if not token:
        print("[ERROR] 找不到 Token")
        return
    
    print(f"Token 前缀: {token[:20]}...")
    print(f"Token 长度: {len(token)} 字符")
    print()
    
    # 检查格式
    if token.startswith("ghp_"):
        print("[OK] Token 类型: Classic PAT")
        print("   需要权限: repo")
    elif token.startswith("github_pat_"):
        print("[OK] Token 类型: Fine-grained PAT")
        print("   需要权限: Contents (Read and write)")
    else:
        print("[WARN] Token 格式异常")
    
    print()
    print("验证步骤:")
    print("1. 访问 https://github.com/settings/tokens")
    print("2. 确认 Token 未过期")
    print("3. 检查权限设置")
    print()
    
    # 尝试 API 请求
    print("测试 API 连接...")
    try:
        import urllib.request
        import json
        
        url = "https://api.github.com/user"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        request = urllib.request.Request(url, headers=headers)
        
        # 检查代理
        proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        if proxy:
            print(f"   使用代理: {proxy}")
        
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"[OK] Token 有效！")
                print(f"   用户名: {data.get('login')}")
            else:
                print(f"[ERROR] HTTP {response.status}")
                
    except Exception as e:
        print(f"[ERROR] 请求失败: {e}")
        print()
        print("可能原因:")
        print("- Token 无效或过期")
        print("- 网络问题")
        print("- 代理设置问题")


if __name__ == "__main__":
    print("="*60)
    print("GitHub Token 诊断工具")
    print("="*60)
    print()
    check_token()
