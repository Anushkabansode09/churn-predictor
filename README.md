# churn-predictor
End-to-end customer churn prediction system built with Python, Random Forest (ROC-AUC: 0.84), and deployed via Streamlit. Predicts churn probability for telecom customers with actionable business recommendations.
# 📉 Customer Churn Prediction System

A machine learning web app that predicts whether a telecom customer is likely to churn, built end-to-end from EDA to deployment.

🔗 **Live App:** https://churn-predictor-m4vzusfbkup98p7jerzq4g.streamlit.app

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges for subscription-based businesses. Losing a customer costs significantly more than retaining one. This project builds a system that identifies at-risk customers before they leave, enabling proactive retention strategies.

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset (via Kaggle)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No) — 26.5% churn rate (imbalanced)

---

## 🔍 Key EDA Findings

- Month-to-month contract customers churn at 3x the rate of annual contract customers
- Customers with tenure under 12 months are the highest churn risk group
- Higher monthly charges correlate strongly with churn
- Customers without online security or tech support churn significantly more

---

## ⚙️ Methodology

| Step | Details |
|------|---------|
| Data Cleaning | Fixed TotalCharges dtype, handled 11 missing values |
| Encoding | LabelEncoder for all categorical features |
| Scaling | StandardScaler on numeric features |
| Imbalance Handling | scale_pos_weight parameter in Random Forest |
| Model | Random Forest Classifier (n_estimators=200, max_depth=10) |
| Evaluation | ROC-AUC, Precision, Recall, F1, Confusion Matrix |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.84** |
| Model | Random Forest |
| Train/Test Split | 80/20 |

---

## 🚀 Deployment

Built with **Streamlit** and deployed on **Streamlit Community Cloud**.

The app takes customer inputs and returns:
- Churn probability (%)
- Risk classification (High / Low)
- Business recommendation

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Random Forest)
- Matplotlib, Seaborn
- Streamlit
- GitHub + Streamlit Cloud

--

includes SHAP-based model explainability — shows which features drive each individual prediction

## 📁 Project Structure
