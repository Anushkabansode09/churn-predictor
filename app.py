import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Page Config ---
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    body { background-color: #0f1117; }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    
    .header-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 5px solid #e94560;
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .header-box h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
    }
    .header-box p {
        color: #a0a0b0;
        font-size: 1rem;
        margin: 5px 0 0 0;
    }
    .section-title {
        color: #e94560;
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 2px solid #e94560;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    .metric-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #c0c0d0 !important;
        font-size: 0.9rem;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        border: none;
        padding: 12px 40px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #c0392b, #e94560);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
with open('rf_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Header ---
st.markdown("""
    <div class="header-box">
        <h1>📉 Customer Churn Predictor</h1>
        <p>Predict churn risk for telecom customers using Machine Learning · Random Forest · ROC-AUC: 0.84</p>
    </div>
""", unsafe_allow_html=True)

# --- Input Section ---
st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

# --- Encode ---
def encode(val, options):
    return options.index(val)

input_data = np.array([[
    encode(gender, ["Female", "Male"]),
    encode(senior, ["No", "Yes"]),
    encode(partner, ["No", "Yes"]),
    encode(dependents, ["No", "Yes"]),
    tenure,
    encode(phone_service, ["No", "Yes"]),
    encode(multiple_lines, ["No", "No phone service", "Yes"]),
    encode(internet_service, ["DSL", "Fiber optic", "No"]),
    encode(online_security, ["No", "No internet service", "Yes"]),
    encode(online_backup, ["No", "No internet service", "Yes"]),
    encode(device_protection, ["No", "No internet service", "Yes"]),
    encode(tech_support, ["No", "No internet service", "Yes"]),
    encode(streaming_tv, ["No", "No internet service", "Yes"]),
    encode(streaming_movies, ["No", "No internet service", "Yes"]),
    encode(contract, ["Month-to-month", "One year", "Two year"]),
    encode(paperless, ["No", "Yes"]),
    encode(payment, ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]),
    monthly_charges,
    total_charges
]])

input_scaled = scaler.transform(input_data)

# --- Predict Button ---
st.markdown("---")
predict_btn = st.button("🔍 Predict Churn Risk")

if predict_btn:
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.markdown("---")
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if pred == 1:
            st.error(f"⚠️ HIGH CHURN RISK — {prob*100:.1f}% probability")
            st.markdown("**Recommended Action:** Offer a discounted annual plan or loyalty reward immediately.")
        else:
            st.success(f"✅ LOW CHURN RISK — {prob*100:.1f}% probability")
            st.markdown("**Status:** Customer is likely to stay. Monitor monthly.")

        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")

    with res_col2:
        # Gauge Chart
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')

        theta = np.linspace(0, np.pi, 200)
        ax.plot(theta, [1]*200, color='#2a2a4a', linewidth=15, alpha=0.5)

        color = '#e94560' if prob > 0.5 else '#2ecc71'
        theta_fill = np.linspace(0, np.pi * prob, 200)
        ax.plot(theta_fill, [1]*200, color=color, linewidth=15)

        ax.set_ylim(0, 1.5)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines.clear()
        ax.text(np.pi/2, 0.3, f"{prob*100:.1f}%", ha='center', va='center',
                fontsize=24, fontweight='bold', color=color,
                transform=ax.transData)
        ax.set_title("Churn Probability Gauge", color='white', pad=10)
        st.pyplot(fig)
        plt.close()

    # Feature Importance Chart
    st.markdown("---")
    st.markdown('<div class="section-title">What\'s Driving This Prediction?</div>', unsafe_allow_html=True)

    feature_names = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure',
                     'Phone Service', 'Multiple Lines', 'Internet Service',
                     'Online Security', 'Online Backup', 'Device Protection',
                     'Tech Support', 'Streaming TV', 'Streaming Movies',
                     'Contract', 'Paperless Billing', 'Payment Method',
                     'Monthly Charges', 'Total Charges']

    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=True).tail(10)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor('#0f1117')
    ax2.set_facecolor('#0f1117')
    bars = ax2.barh(feat_df['Feature'], feat_df['Importance'], color='#e94560', alpha=0.85)
    ax2.set_xlabel('Importance Score', color='white')
    ax2.set_title('Top 10 Features Driving Churn', color='white', fontsize=13)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#2a2a4a')
    ax2.spines['left'].set_color('#2a2a4a')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    st.pyplot(fig2)
    plt.close()
