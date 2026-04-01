# ESA Resistance Risk Predictor

This repository contains Streamlit-based web calculators for predicting the risk of next-quarter ESA resistance.

## Available apps

- `eri_q4_streamlit_app.py`: legacy app entry point
- `eri_q4_streamlit_app_en.py`: English academic-style app with compact and full calculators

## Local run

```bash
pip install -r requirements.txt
streamlit run eri_q4_streamlit_app_en.py
```

## Deployment

The project can be deployed on Streamlit Community Cloud.

1. Push the repository to GitHub.
2. Open https://share.streamlit.io/
3. Select the repository.
4. Set the main file path to `eri_q4_streamlit_app_en.py` for the English version.
5. Click `Deploy`.

## Automated tests

```bash
python -m pytest tests/test_streamlit_prediction_e2e_en.py -q
```
