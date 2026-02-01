#!/usr/bin/env python3
"""
GitHub 推送助手 (安全版)
=======================

这个脚本帮助你安全地将代码推送到 GitHub，
Token 从环境变量读取，不会暴露在代码中。

使用方法:
    python push_to_github.py

前置条件:
    1. 设置环境变量 GITHUB_TOKEN (见下方说明)
    2. 在 GitHub 上创建好仓库
    3. 本地 Git 仓库已初始化
"""

import os
import sys
import subprocess
from pathlib import Path


def get_github_token():
    """
    从环境变量或 .env 文件获取 GitHub Token。
    
    安全提示:
        - Token 永远不会保存在代码中
        - Token 不会被打印到屏幕
        - Token 不会被记录到日志
    
    设置方法 (按优先级):
        方法 1 - 环境变量:
            $env:GITHUB_TOKEN = "ghp_xxxxxxxx"
        
        方法 2 - .env 文件 (推荐):
            创建 .env 文件，内容: GITHUB_TOKEN=ghp_xxxxxxxx
            (确保 .env 在 .gitignore 中！)
        
        方法 3 - 永久设置:
            [Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "ghp_xxxxxxxx", "User")
    """
    # 尝试 1: 从环境变量读取
    token = os.getenv("GITHUB_TOKEN")
    source = "环境变量"
    
    # 尝试 2: 从 .env 文件读取
    if not token and Path(".env").exists():
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 跳过注释和空行
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("GITHUB_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        token = token.strip('"\'')  # 去除引号
                        source = ".env 文件"
                        break
        except Exception as e:
            print(f"⚠️  读取 .env 文件失败: {e}")
    
    if not token:
        print("❌ 错误: 找不到 GITHUB_TOKEN！")
        print("\n解决方法（选择一种）:")
        print("\n方法 1 - 临时设置 (立即生效):")
        print("  $env:GITHUB_TOKEN = '你的token'")
        print("\n方法 2 - .env 文件 (推荐):")
        print("  1. 确保 .env 文件存在")
        print("  2. 内容: GITHUB_TOKEN=你的token")
        print("  3. 确保 .env 在 .gitignore 中")
        print("\n方法 3 - 永久设置:")
        print("  [Environment]::SetEnvironmentVariable('GITHUB_TOKEN', '你的token', 'User')")
        print("  然后重启 PowerShell")
        print("\n获取 Token: https://github.com/settings/tokens")
        sys.exit(1)
    
    print(f"   ✅ 从 {source} 读取到 Token")
    
    # 验证 Token 格式 (简单检查)
    if not token.startswith(("ghp_", "github_pat_")):
        print("⚠️  警告: Token 格式看起来不对，应该以 ghp_ 或 github_pat_ 开头")
        confirm = input("是否继续? (yes/no): ")
        if confirm.lower() != "yes":
            sys.exit(0)
    
    return token


def get_git_info():
    """
    获取 Git 仓库信息。
    
    Returns:
        dict: 包含用户名、仓库名、当前分支
    """
    # GitHub 用户名（修改这里）
    github_username = "li-guohao"
    
    # 仓库名（修改这里，如果你的仓库名不同）
    repo_name = "asam-attention"
    
    # 获取当前分支名
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = result.stdout.strip()
    except subprocess.CalledProcessError:
        current_branch = "main"  # 默认分支
    
    return {
        "username": github_username,
        "repo": repo_name,
        "branch": current_branch
    }


def check_git_status():
    """
    检查 Git 仓库状态。
    
    Returns:
        bool: 是否有未提交的更改
    """
    try:
        # 检查是否有未提交的更改
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            print("⚠️ 警告: 有未提交的更改！")
            print("\n请先提交更改:")
            print("  git add .")
            print("  git commit -m '你的提交信息'")
            print("\n或者使用:")
            print("  git status  # 查看详细状态")
            
            choice = input("\n是否继续推送? (yes/no): ")
            return choice.lower() == "yes"
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git 状态检查失败: {e}")
        return False


def setup_remote(token, username, repo):
    """
    配置 Git 远程仓库。
    
    Args:
        token: GitHub Personal Access Token
        username: GitHub 用户名
        repo: 仓库名
    
    Returns:
        bool: 是否成功
    """
    # 构建带认证的远程 URL
    # 格式: https://用户名:Token@github.com/用户名/仓库.git
    remote_url = f"https://{username}:{token}@github.com/{username}/{repo}.git"
    
    try:
        # 检查是否已有 remote
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 已有 remote，更新 URL
            print(f"📝 更新远程仓库地址...")
            subprocess.run(
                ["git", "remote", "set-url", "origin", remote_url],
                check=True
            )
        else:
            # 没有 remote，添加
            print(f"📝 添加远程仓库...")
            subprocess.run(
                ["git", "remote", "add", "origin", remote_url],
                check=True
            )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 配置远程仓库失败: {e}")
        return False


def push_to_github(branch):
    """
    推送代码到 GitHub。
    
    Args:
        branch: 分支名
    
    Returns:
        bool: 是否成功
    """
    print(f"🚀 推送到 GitHub (分支: {branch})...")
    print("  这可能需要几秒到几分钟...\n")
    
    try:
        # 执行推送
        result = subprocess.run(
            ["git", "push", "-u", "origin", branch],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 输出推送结果
        if result.stdout:
            print(result.stdout)
        
        print("✅ 推送成功！")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 推送失败！")
        print(f"\n错误信息:\n{e.stderr}")
        
        # 常见错误提示
        if "rejected" in e.stderr.lower():
            print("\n💡 提示: 远程仓库有更新，请先拉取:")
            print("  git pull origin main --rebase")
        elif "could not resolve" in e.stderr.lower():
            print("\n💡 提示: 网络连接问题，请检查网络")
        elif "authentication" in e.stderr.lower():
            print("\n💡 提示: Token 可能无效或过期，请检查:")
            print("  https://github.com/settings/tokens")
        
        return False


def main():
    """
    主函数：协调整个推送流程。
    """
    print("="*60)
    print("GitHub 推送助手 (安全版)")
    print("="*60)
    print()
    
    # 1. 获取 Token（安全读取）
    print("🔐 步骤 1: 读取 GitHub Token...")
    token = get_github_token()
    print("   ✅ Token 已读取（已隐藏）\n")
    
    # 2. 获取仓库信息
    print("🔍 步骤 2: 获取仓库信息...")
    info = get_git_info()
    print(f"   用户名: {info['username']}")
    print(f"   仓库名: {info['repo']}")
    print(f"   分支:   {info['branch']}\n")
    
    # 3. 检查 Git 状态
    print("📋 步骤 3: 检查 Git 状态...")
    if not check_git_status():
        sys.exit(1)
    print("   ✅ Git 状态正常\n")
    
    # 4. 配置远程仓库
    print("⚙️  步骤 4: 配置远程仓库...")
    if not setup_remote(token, info['username'], info['repo']):
        sys.exit(1)
    print("   ✅ 远程仓库配置完成\n")
    
    # 5. 推送到 GitHub
    print("📤 步骤 5: 推送到 GitHub...")
    if not push_to_github(info['branch']):
        sys.exit(1)
    
    # 完成
    print()
    print("="*60)
    print("🎉 推送完成！")
    print("="*60)
    print(f"\n访问你的仓库:")
    print(f"  https://github.com/{info['username']}/{info['repo']}")
    print()
    print("建议下一步:")
    print("  1. 在 GitHub 上创建 Release")
    print("  2. 添加 Topics 标签")
    print("  3. 分享你的项目！")
    print()


if __name__ == "__main__":
    # 安全检查：确保当前目录是 Git 仓库
    if not Path(".git").exists():
        print("❌ 错误: 当前目录不是 Git 仓库！")
        print("\n请切换到项目目录:")
        print("  cd e:\\GIT\\asam-attention")
        sys.exit(1)
    
    main()
