param(
    [string]$RepoName = "esa-risk-predictor"
)

$ErrorActionPreference = "Stop"
$git = "C:\Program Files\Git\cmd\git.exe"
$gh = "C:\Program Files\GitHub CLI\gh.exe"

if (-not (Test-Path $git)) {
    throw "Git not found at $git"
}

if (-not (Test-Path $gh)) {
    throw "GitHub CLI not found at $gh"
}

& $gh auth status | Out-Null

if (-not (Test-Path ".git")) {
    & $git init
    & $git branch -M main
}

& $git add .gitignore
& $git add .streamlit/config.toml
& $git add README.md
& $git add requirements.txt
& $git add eri_q4_streamlit_app.py
& $git add tests/test_streamlit_prediction_e2e.py
& $git add outputs/eri_q4_paper/artifacts

$hasCommit = (& $git rev-parse --verify HEAD 2>$null)
if (-not $hasCommit) {
    & $git -c user.name="Codex Local" -c user.email="codex-local@example.com" commit -m "Prepare Streamlit app for cloud deployment"
} else {
    & $git -c user.name="Codex Local" -c user.email="codex-local@example.com" commit -m "Update Streamlit deployment bundle"
}

& $gh repo create $RepoName --private --source . --remote origin --push

Write-Host ""
Write-Host "GitHub push completed."
Write-Host "Next step: open https://share.streamlit.io/ and deploy main file eri_q4_streamlit_app.py"
