import os
import streamlit as st
import requests

# Detect environment (Docker or Local)
API_BASE_URL = "http://api:8000" if os.getenv("DOCKER") else "http://127.0.0.1:8000"

st.set_page_config(page_title="Candidate Evaluation", layout="centered")
st.title("Recruitment Prediction App")
st.markdown("Enter the candidate's experience and technical test score to predict hiring decision.")

# Input fields
experience = st.number_input("Experience (years)", min_value=0.0, max_value=10.0, step=0.5, format="%.1f")
score = st.number_input("Technical Score (0 - 100)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f")

# Predict button
if st.button("Predict"):
    try:
        response = requests.post(
            url=f"{API_BASE_URL}/predict",
            json={"experience_years": experience, "technical_score": score}
        )

        if response.status_code == 200:
            result = response.json()
            if "Rejected" in result["prediction"]:
                st.error(f"Prediction: {result['prediction']}")
            else:
                st.success(f"Prediction: {result['prediction']}")
        else:
            st.warning("Unexpected response from the prediction API.")

    except Exception as e:
        st.error(f"Error connecting to the prediction service: {e}")
