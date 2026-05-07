# 📡 Telco Customer Churn Predictor

An interactive Streamlit app that runs the full churn prediction pipeline:
**EDA → Feature Engineering → Random Forest → Evaluation → Live Prediction**

## 🚀 Deploy on Streamlit Cloud

1. Push this folder to a **GitHub repository** (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo, branch, and set the main file to `app.py`.
4. Click **Deploy** — it will install `requirements.txt` automatically.

## 📦 Dataset

Upload **`WA_Fn-UseC_-Telco-Customer-Churn.csv`** inside the app sidebar.  
Download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## 🗂 File Structure

```
├── app.py            ← Main Streamlit app
├── requirements.txt  ← Python dependencies
└── README.md
```

## 🧠 Model

| Setting | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 100 |
| Max depth | 10 |
| Class weight | balanced |
| Test split | 20% |
