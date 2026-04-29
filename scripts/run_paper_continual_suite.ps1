$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$ForwardArgs = @($Args)

if ($ForwardArgs -notcontains "--paper-tex") {
    $ForwardArgs += @("--paper-tex", "paper/asam_paper.tex")
}

if ($ForwardArgs -notcontains "--appendix-only-tex") {
    $ForwardArgs += @("--appendix-only-tex", "paper/continual_appendix_only.tex")
}

python scripts/run_continual_paper_suite.py @ForwardArgs
