# ESA Resistance Risk Predictor

这是一个基于 Streamlit 的 ESA 抵抗风险预测应用，入口文件是 `eri_q4_streamlit_app.py`。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run eri_q4_streamlit_app.py
```

## 固定线上部署

推荐使用 Streamlit Community Cloud，部署后会得到一个长期稳定的 `https://<your-app>.streamlit.app` 链接。

### 1. 上传到 GitHub

- 新建一个 GitHub 仓库
- 把本项目上传到仓库
- 确认以下文件已经在仓库中：
  - `eri_q4_streamlit_app.py`
  - `requirements.txt`
  - `outputs/eri_q4_paper/artifacts/`

### 2. 在 Streamlit Community Cloud 部署

- 打开 https://share.streamlit.io/
- 用 GitHub 账号登录
- 选择你的仓库
- Main file path 填 `eri_q4_streamlit_app.py`
- 点击 Deploy

### 3. 部署完成后

- 平台会分配一个固定网址
- 以后只要 GitHub 仓库还在、应用未被手动删除，这个网址通常会保持不变

## 自动化测试

```bash
python -m pytest tests/test_streamlit_prediction_e2e.py -q
```
