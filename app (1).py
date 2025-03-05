 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Load the trained model
model = joblib.load("xgboost_model.pkl")  # Ensure this file exists
scaler = joblib.load("scaler.pkl")  # Load the feature scaler

# Streamlit UI
st.title("🚑 Insurance Claim Prediction App")
st.markdown("### Enter customer details to predict whether they will file an insurance claim.")

# User Inputs
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10, 50, 25)
children = st.slider("Number of Children", 0, 5, 1)
charges = st.number_input("Medical Charges", min_value=1000, max_value=100000, value=5000)
sex = st.radio("Sex", ["Male", "Female"])
smoker = st.radio("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Encode categorical values
sex_encoded = 1 if sex == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0

# One-hot encode region
region_dict = {"Northeast": [1, 0, 0], "Northwest": [0, 1, 0], "Southeast": [0, 0, 1], "Southwest": [0, 0, 0]}
region_encoded = region_dict[region]

# Create input DataFrame
input_data = pd.DataFrame([[age, bmi, children, charges, sex_encoded, smoker_encoded] + region_encoded],
                          columns=["age", "bmi", "children", "charges", "sex", "smoker", "region_northeast", "region_northwest", "region_southeast"])

# Scale the features
input_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display Results
    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk: The person is **likely to file an insurance claim**. (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk: The person is **not likely to file an insurance claim**. (Confidence: {1 - probability:.2f})")

    # Feature Importance
    st.subheader("🔑 Feature Importance")
    feature_importance = pd.Series(model.feature_importances_, index=input_data.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
    plt.title("Feature Importance")
    st.pyplot(fig)
