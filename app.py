import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Delivery Time Estimator", layout="centered")

# ----------------------------
# Load trained model + encoder
# ----------------------------
model = joblib.load(r"C:\Users\Mitali\OneDrive\Desktop\Python WorkSpace\trained_model.pkl")
encoder = joblib.load(r"C:\Users\Mitali\OneDrive\Desktop\Python WorkSpace\label_encoders.pkl")

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📦 Delivery Time Estimator")
st.write("Enter details below to estimate delivery time")

# User Inputs
weather = st.selectbox("Weather", encoder["Weather"].classes_)
traffic = st.selectbox("Traffic", encoder["Traffic"].classes_)
vehicle = st.selectbox("Vehicle", encoder["Vehicle"].classes_)
category = st.selectbox("Category", encoder["Category"].classes_)
area = st.selectbox("Area", encoder["Area"].classes_)

agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0, 0.1)
order_time_hr = st.slider("Order Time (Hour)", 0.0, 23.9, 12.0, 0.5)
pickup_time_hr = st.slider("Pickup Time (Hour)", 0.0, 23.9, 13.0, 0.5)

# ----------------------------
# Prepare Input Row
# ----------------------------
input_dict = {
    "Weather": encoder["Weather"].transform([weather])[0],
    "Traffic": encoder["Traffic"].transform([traffic])[0],
    "Vehicle": encoder["Vehicle"].transform([vehicle])[0],
    "Category": encoder["Category"].transform([category])[0],
    "Area": encoder["Area"].transform([area])[0],
    "Agent_Age": agent_age,
    "Agent_Rating": agent_rating,
    "Order_Time_hr": order_time_hr,
    "Pickup_Time_hr": pickup_time_hr
}

input_df = pd.DataFrame([input_dict])

st.write("### Encoded Input Summary")
st.dataframe(input_df)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Estimate Delivery Time"):
    st.write(f"Model expects {model.n_features_in_} features")
    st.write(f"Input has {input_df.shape[1]} features")

    if input_df.shape[1] == model.n_features_in_:
        pred = model.predict(input_df)[0]
        st.metric("Estimated Delivery Time (minutes)", f"{pred:.2f}")
    else:
        st.error(
            f"❌ Feature mismatch: model expects {model.n_features_in_} features, but got {input_df.shape[1]}.\n"
            "Please ensure all required inputs are included."
        )
