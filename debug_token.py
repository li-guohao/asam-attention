#!/usr/bin/env python3
"""
Token 调试工具 - 检查 Token 是否正确设置
"""

import os
from pathlib import Path

print("="*60)
print("Token Debug Info")
print("="*60)
print()

# 检查环境变量
env_token = os.getenv("GITHUB_TOKEN")
print("1. Environment Variable GITHUB_TOKEN:")
if env_token:
    print("   [OK] Set")
    print(f"   Prefix: {env_token[:15]}...")
    print(f"   Length: {len(env_token)} chars")
    if env_token.startswith('ghp_'):
        print("   Format: Classic PAT (correct)")
    elif env_token.startswith('github_pat_'):
        print("   Format: Fine-grained PAT")
    else:
        print("   Format: Unknown (may be wrong)")
else:
    print("   [NOT SET]")

print()

# 检查 .env 文件
print("2. .env File:")
if Path(".env").exists():
    print("   [OK] File exists")
    with open(".env", "r") as f:
        content = f.read()
        if "GITHUB_TOKEN=" in content:
            for line in content.split("\n"):
                if line.startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"\'')
                    if token and token != "your_token_here" and token != "github_pat_xxx":
                        print("   [OK] Token configured")
                        print(f"   Prefix: {token[:15]}...")
                        print(f"   Length: {len(token)} chars")
                    else:
                        print("   [ERROR] Token is placeholder")
                    break
        else:
            print("   [ERROR] GITHUB_TOKEN not found")
else:
    print("   [ERROR] File not found")

print()
print("="*60)
print("Possible Issues:")
print("="*60)
print()

if env_token:
    print("Environment variable is set")
    print("But if you just set it, you need to RESTART PowerShell")
    print()
    print("Quick fix (immediate):")
    print("  $env:GITHUB_TOKEN = 'ghp_your_actual_token'")
    print("  python tools/github-setup/push_to_github.py")
else:
    print("Environment variable NOT set")
    print()
    print("Solutions:")
    print("  1. Temporary: $env:GITHUB_TOKEN = 'ghp_your_token'")
    print("  2. Edit .env file with your token")
