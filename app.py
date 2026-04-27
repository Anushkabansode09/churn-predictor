import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('rf_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

st.title("📉 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn risk.")

# --- Input Fields ---
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

with col2:
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

# --- Encoding (must match training encoding) ---
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

# --- Prediction ---
st.markdown("---")
if st.button("🔍 Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error(f"⚠️ High Churn Risk — {prob*100:.1f}% probability")
        st.markdown("**Recommended Action:** Offer this customer a discounted annual plan or loyalty reward immediately.")
    else:
        st.success(f"✅ Low Churn Risk — {prob*100:.1f}% probability")
        st.markdown("**Status:** Customer is likely to stay. Monitor monthly.")

    st.metric("Churn Probability", f"{prob*100:.1f}%")
    