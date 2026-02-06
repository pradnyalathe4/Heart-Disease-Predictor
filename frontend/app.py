import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.center-title {
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 15px;
}

button[kind="primary"] {
    padding: 0.35rem 0.9rem;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="center-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["📁 Dataset", "🧑‍⚕️ Prediction", "ℹ️ About"])

# ================= DATASET TAB =================
with tabs[0]:
    st.subheader("📊 Model Training & Evaluation")

    st.markdown("### Train Model")
    train_file = st.file_uploader("Training Dataset (CSV)", type="csv")

    if st.button("Train"):
        if train_file:
            r = requests.post(f"{API}/train", files={"file": train_file})
            st.success(r.json()["message"])
        else:
            st.warning("Please upload training dataset")

    st.divider()

    st.markdown("### Evaluate Model")
    test_file = st.file_uploader("Testing Dataset (CSV)", type="csv")

    if st.button("Evaluate"):
        if test_file:
            r = requests.post(f"{API}/evaluate", files={"file": test_file})
            res = r.json()

            st.metric("Accuracy", round(res["accuracy"], 2))
            st.dataframe(pd.DataFrame(res["report"]))
        else:
            st.warning("Please upload testing dataset")

with tabs[1]:
    st.subheader("🧑‍⚕️ Patient Information")

    st.markdown("**Basic Measurements**")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 100, step=1)
        ex = st.number_input("Exercise Hours", step=1)

    with c2:
        hr = st.number_input("Heart Rate", step=1)
        chol = st.number_input("Cholesterol", step=1)
        stress = st.number_input("Stress Level", step=1)

    with c3:
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
        sugar = st.number_input("Blood Sugar", step=1)
        bp = st.number_input(
            "Blood Pressure (mm Hg)",
            step=1
        )

    st.divider()

    st.markdown("**Medical & Lifestyle Details**")
    c4, c5, c6 = st.columns(3)

    with c4:
        angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        family = st.selectbox("Family History", ["No", "Yes"])

    with c5:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])

    with c6:
        chest = st.selectbox(
            "Chest Pain Type",
            ["Atypical Angina", "Typical Angina", "Non-anginal Pain", "Asymptomatic"]
        )
        alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])

    st.divider()

    left, right = st.columns([1, 3])

    with left:
        predict = st.button("🔍 Predict")

    with right:
        result_box = st.empty()

    if predict:
        payload = {
            "Age": age,
            "Cholesterol": chol,
            "Blood_Pressure": bp,
            "Heart_Rate": hr,
            "Exercise_Hours": ex,
            "Stress_Level": stress,
            "Blood_Sugar": sugar,
            "Gender": gender,
            "Alcohol_Intake": alcohol,
            "Family_History": family,
            "Diabetes": diabetes,
            "Obesity": obesity,
            "Exercise_Induced_Angina": angina,
            "Smoking": smoking,
            "Chest_Pain_Type": chest
        }

        r = requests.post(f"{API}/predict", json=payload)
        res = r.json()

        if "prediction" not in res:
            result_box.error("Backend error")
        else:
            confidence = res["confidence"] * 100

            if res["prediction"] == 1:
                result_box.error(
                    f"❤️ High Risk of Heart Disease\n\nConfidence: {confidence:.1f}%"
                )
            else:
                result_box.success(
                    f"✅ Low Risk of Heart Disease\n\nConfidence: {confidence:.1f}%"
                )

with tabs[2]:
    st.markdown("""
### ℹ️ About Project

**Heart Disease Prediction System**

A clean, compact and industry-grade ML application.

**Tech Stack**
- FastAPI
- Streamlit
- Random Forest
- Scikit-learn
""")

