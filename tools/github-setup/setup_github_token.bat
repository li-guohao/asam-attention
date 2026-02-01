@echo off
chcp 65001 >nul
echo ========================================
echo GitHub Token 配置助手
echo ========================================
echo.
echo 这个脚本会设置 GITHUB_TOKEN 环境变量
echo.
echo 请前往以下链接生成 Token:
echo   https://github.com/settings/tokens
echo.
echo 需要的权限:
echo   [x] repo (访问仓库)
echo.

set /p TOKEN="请输入你的 GitHub Token: "

if "%TOKEN%"=="" (
    echo 错误: Token 不能为空！
    pause
    exit /b 1
)

:: 保存到环境变量（用户级别）
setx GITHUB_TOKEN "%TOKEN%"

echo.
echo ========================================
echo ✅ Token 已保存！
echo ========================================
echo.
echo 使用方法:
echo   python push_to_github.py
echo.
echo 注意: 如果这是第一次设置，请重启 PowerShell 或 CMD
echo.
pause
