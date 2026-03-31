# Long-Term Deployment Steps

## 1. 登录 GitHub CLI

```bash
gh auth login
```

推荐选择：

- GitHub.com
- HTTPS
- Login with a web browser

## 2. 自动创建仓库并推送

```powershell
.\publish_to_github.ps1 -RepoName "esa-risk-predictor"
```

## 3. 在 Streamlit Community Cloud 生成固定网址

打开 [https://share.streamlit.io/](https://share.streamlit.io/)：

- 选择刚创建的 GitHub 仓库
- Branch: `main`
- Main file path: `eri_q4_streamlit_app.py`
- 点击 `Deploy`

部署成功后会生成一个长期可用的固定网址，通常形如：

`https://<your-app-name>.streamlit.app`
