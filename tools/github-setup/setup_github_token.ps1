# GitHub Token 配置脚本
# =====================
# 
# 这个脚本帮助你在 Windows 上安全地配置 GitHub Token
# Token 会被保存在系统环境变量中，其他程序可以读取
#
# 使用方法:
#   1. 打开 PowerShell
#   2. 运行: .\setup_github_token.ps1
#   3. 按提示输入你的 Token

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Token 配置助手" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查当前是否有 Token
$existingToken = [Environment]::GetEnvironmentVariable("GITHUB_TOKEN", "User")

if ($existingToken) {
    Write-Host "⚠️  警告: 已存在 GITHUB_TOKEN 环境变量" -ForegroundColor Yellow
    Write-Host "   当前值: $($existingToken.Substring(0, 8))..." -ForegroundColor Gray
    Write-Host ""
    
    $replace = Read-Host "是否替换? (y/n)"
    if ($replace -ne "y") {
        Write-Host ""
        Write-Host "已取消，保留现有 Token" -ForegroundColor Green
        exit 0
    }
}

Write-Host "📋 请前往以下链接生成 Token:" -ForegroundColor Cyan
Write-Host "   https://github.com/settings/tokens" -ForegroundColor Blue
Write-Host ""
Write-Host "需要的权限:" -ForegroundColor Yellow
Write-Host "   ☑️ repo (访问仓库)"
Write-Host ""
Write-Host "⚠️  安全提示:" -ForegroundColor Red
Write-Host "   - Token 就像密码，不要分享给他人"
Write-Host "   - 不要将 Token 写入代码文件"
Write-Host "   - 定期更换 Token 以提高安全性"
Write-Host ""

# 读取 Token（输入时隐藏）
$token = Read-Host -Prompt "请输入你的 GitHub Token" -AsSecureString

# 将 SecureString 转换为明文（仅用于验证和保存）
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($token)
$tokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# 验证 Token 格式
if (-not ($tokenPlain -match "^(ghp_|github_pat_)")) {
    Write-Host ""
    Write-Host "❌ 错误: Token 格式不正确！" -ForegroundColor Red
    Write-Host "   Token 应该以 'ghp_' 或 'github_pat_' 开头" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "请重新运行脚本并输入正确的 Token" -ForegroundColor Cyan
    exit 1
}

# 设置环境变量（用户级别，永久有效）
[Environment]::SetEnvironmentVariable("GITHUB_TOKEN", $tokenPlain, "User")

Write-Host ""
Write-Host "✅ 成功！Token 已保存到系统环境变量" -ForegroundColor Green
Write-Host ""
Write-Host "验证方法:" -ForegroundColor Cyan
Write-Host "   1. 重启 PowerShell"
Write-Host "   2. 运行: `$env:GITHUB_TOKEN"
Write-Host "   应该显示你的 Token（开头部分）"
Write-Host ""
Write-Host "使用方法:" -ForegroundColor Cyan
Write-Host "   python push_to_github.py"
Write-Host ""

# 清理内存中的明文 Token
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
$tokenPlain = $null
