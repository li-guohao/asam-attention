#!/usr/bin/env python3
"""
GitHub 配置检查工具
===================

检查你的 GitHub 配置是否正确，无需担心 Token 泄露。

使用方法:
    python check_setup.py

检查内容:
    1. 环境变量 GITHUB_TOKEN 是否设置
    2. Token 格式是否正确
    3. 是否能连接到 GitHub
    4. Git 仓库配置是否正确
"""

import os
import sys
import subprocess
from pathlib import Path


def check_color_support():
    """检查终端是否支持颜色输出。"""
    return sys.platform != "win32" or "ANSICON" in os.environ or "WT_SESSION" in os.environ


def print_success(message):
    """打印成功信息（绿色）。"""
    if check_color_support():
        print(f"\033[32m✅ {message}\033[0m")
    else:
        print(f"[OK] {message}")


def print_error(message):
    """打印错误信息（红色）。"""
    if check_color_support():
        print(f"\033[31m❌ {message}\033[0m")
    else:
        print(f"[ERROR] {message}")


def print_warning(message):
    """打印警告信息（黄色）。"""
    if check_color_support():
        print(f"\033[33m⚠️  {message}\033[0m")
    else:
        print(f"[WARNING] {message}")


def print_info(message):
    """打印普通信息。"""
    print(f"   {message}")


def check_github_token():
    """
    检查 GitHub Token 是否设置。
    
    Returns:
        tuple: (是否成功, Token值)
    """
    print("\n" + "="*60)
    print("步骤 1: 检查 GitHub Token")
    print("="*60)
    
    # 从环境变量读取
    token = os.getenv("GITHUB_TOKEN")
    
    # 如果环境变量没有，尝试从 .env 文件读取
    if not token and Path(".env").exists():
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GITHUB_TOKEN=") and not line.startswith("#"):
                        token = line.split("=", 1)[1].strip()
                        token = token.strip('"\'')  # 去除引号
                        print_success("从 .env 文件读取到 Token")
                        break
        except Exception:
            pass
    
    if not token:
        print_error("找不到 GITHUB_TOKEN！")
        print_info("")
        print_info("解决方法（选择一种）:")
        print_info("")
        print_info("方法 1 - 临时设置 (立即生效):")
        print_info("  $env:GITHUB_TOKEN = '你的token'")
        print_info("")
        print_info("方法 2 - .env 文件 (推荐):")
        print_info("  1. 复制 .env.example 为 .env")
        print_info("  2. 编辑 .env，填入你的 Token")
        print_info("  3. 确保 .env 在 .gitignore 中")
        print_info("")
        print_info("方法 3 - 永久设置:")
        print_info("  运行: setup_github_token.bat")
        print_info("  然后重启 PowerShell")
        return False, None
    
    # Token 存在，显示部分信息（安全）
    print_success("找到 GITHUB_TOKEN 环境变量")
    print_info(f"Token 前缀: {token[:8]}...")
    print_info(f"Token 长度: {len(token)} 字符")
    
    # 检查格式
    if token.startswith("ghp_"):
        print_success("Token 格式正确 (Classic)")
    elif token.startswith("github_pat_"):
        print_success("Token 格式正确 (Fine-grained)")
    else:
        print_warning("Token 格式异常 (不以 ghp_ 或 github_pat_ 开头)")
        print_info("这可能不是有效的 GitHub Token")
    
    return True, token


def check_git_repo():
    """
    检查 Git 仓库配置。
    
    Returns:
        bool: 是否通过检查
    """
    print("\n" + "="*60)
    print("步骤 2: 检查 Git 仓库")
    print("="*60)
    
    # 检查是否是 Git 仓库
    if not Path(".git").exists():
        print_error("当前目录不是 Git 仓库！")
        print_info("")
        print_info("解决方法:")
        print_info("  1. 确保在项目根目录运行此脚本")
        print_info("  2. 或者先初始化 Git 仓库:")
        print_info("     git init")
        print_info("     git add .")
        print_info("     git commit -m 'Initial commit'")
        return False
    
    print_success("当前目录是 Git 仓库")
    
    # 检查远程仓库
    try:
        result = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            print_success("已配置远程仓库 (origin)")
            print_info("远程地址:")
            for line in result.stdout.strip().split("\n")[:2]:
                print_info(f"  {line}")
        else:
            print_warning("未配置远程仓库")
            print_info("运行 push_to_github.py 时会自动配置")
        
    except subprocess.CalledProcessError:
        print_warning("无法读取远程仓库配置")
    
    # 检查分支
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()
        print_success(f"当前分支: {branch}")
        
    except subprocess.CalledProcessError:
        print_warning("无法获取当前分支")
    
    # 检查未提交的更改
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            print_warning("有未提交的更改")
            print_info("提交前请先执行:")
            print_info("  git add .")
            print_info("  git commit -m '提交信息'")
        else:
            print_success("工作区干净（无未提交更改）")
        
    except subprocess.CalledProcessError:
        print_warning("无法检查 Git 状态")
    
    return True


def check_github_connection(token):
    """
    测试是否能连接到 GitHub。
    
    Args:
        token: GitHub Token
    
    Returns:
        bool: 是否连接成功
    """
    print("\n" + "="*60)
    print("步骤 3: 测试 GitHub 连接")
    print("="*60)
    
    if not token:
        print_warning("跳过（没有 Token）")
        return False
    
    print("正在连接 GitHub...")
    
    try:
        # 使用 curl 或 python 测试连接
        import urllib.request
        import json
        
        # 构建请求
        url = "https://api.github.com/user"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ASAM-Check-Setup"
        }
        
        request = urllib.request.Request(url, headers=headers)
        
        # 发送请求（设置超时）
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                username = data.get("login", "Unknown")
                print_success(f"成功连接到 GitHub！")
                print_info(f"用户名: {username}")
                print_info(f"ID: {data.get('id', 'N/A')}")
                return True
            else:
                print_error(f"连接失败 (HTTP {response.status})")
                return False
                
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print_error("Token 无效或已过期！")
            print_info("")
            print_info("解决方法:")
            print_info("  1. 访问 https://github.com/settings/tokens")
            print_info("  2. 检查 Token 是否过期")
            print_info("  3. 生成新的 Token")
            print_info("  4. 重新运行 setup_github_token.bat")
        else:
            print_error(f"HTTP 错误: {e.code}")
        return False
        
    except urllib.error.URLError as e:
        print_error("网络连接失败")
        print_info(f"原因: {e.reason}")
        print_info("")
        print_info("可能原因:")
        print_info("  - 无网络连接")
        print_info("  - 代理设置问题")
        print_info("  - 防火墙阻挡")
        return False
        
    except Exception as e:
        print_error(f"检查失败: {str(e)}")
        return False


def print_summary(token_ok, repo_ok, connection_ok):
    """
    打印检查总结。
    
    Args:
        token_ok: Token 检查是否通过
        repo_ok: 仓库检查是否通过
        connection_ok: 连接检查是否通过
    """
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)
    print()
    
    if token_ok and repo_ok and connection_ok:
        print_success("所有检查通过！可以推送代码了！")
        print()
        print("下一步:")
        print("  python push_to_github.py")
        print()
        
    elif token_ok and repo_ok and not connection_ok:
        print_warning("基本配置正确，但无法连接 GitHub")
        print()
        print("可能原因:")
        print("  - 网络问题")
        print("  - Token 权限不足")
        print("  - GitHub 服务异常")
        print()
        print("你可以尝试:")
        print("  python push_to_github.py")
        print()
        
    elif not token_ok:
        print_error("缺少 GitHub Token，无法推送")
        print()
        print("请先设置 Token:")
        print("  setup_github_token.bat")
        print()
        
    else:
        print_warning("配置未完成，请查看上方错误信息")
        print()


def main():
    """主函数：执行所有检查。"""
    print()
    print("="*60)
    print("GitHub 配置检查工具")
    print("="*60)
    print()
    print("这个工具会检查你的配置是否正确，")
    print("不会泄露你的 Token。")
    print()
    
    # 执行检查
    token_ok, token = check_github_token()
    repo_ok = check_git_repo()
    connection_ok = check_github_connection(token) if token_ok else False
    
    # 打印总结
    print_summary(token_ok, repo_ok, connection_ok)


if __name__ == "__main__":
    main()
