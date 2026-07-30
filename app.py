
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #334155 100%
    );
}

/* Text */
html, body, [class*="css"] {
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Titles */
.main-title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    backdrop-filter: blur(10px);
}

[data-testid="stMetricLabel"] {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
}

/* Success Card */
.success-box {
    background: rgba(34,197,94,0.15);
    border: 2px solid #22c55e;
    color: #86efac;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

/* Danger Card */
.danger-box {
    background: rgba(239,68,68,0.15);
    border: 2px solid #ef4444;
    color: #fca5a5;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("❤️ Heart Disease App")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "About"]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown(
        """
        <h1 class="main-title">
        ❤️ Heart Disease Prediction System
        </h1>

        <p class="subtitle">
        AI Powered Heart Disease Risk Assessment
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "KNN")

    with col2:
        st.metric("Features", "11")

    with col3:
        st.metric("Status", "Active")

    st.info(
        "Predict heart disease risk using patient health indicators and machine learning."
    )

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Prediction":

    st.header("🩺 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 100, 45)

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            80, 220, 120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            100, 600, 200
        )

    with col2:

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.slider(
            "Maximum Heart Rate",
            60, 220, 150
        )

        exercise_angina = st.selectbox(
            "Exercise Angina",
            ["Y", "N"]
        )

        oldpeak = st.slider(
            "Oldpeak",
            0.0, 6.0, 1.0
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    if st.button("❤️ Predict Heart Disease"):

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        st.write("")

        if prediction == 1:

            st.markdown(
                """
                <div class="danger-box">
                ⚠️ High Risk of Heart Disease
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="success-box">
                ✅ Low Risk of Heart Disease
                </div>
                """,
                unsafe_allow_html=True
            )

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.title("🤖 About The Model")

    st.success("K-Nearest Neighbors (KNN)")

    st.markdown("""
### Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

### Purpose

This machine learning application predicts the likelihood of heart disease based on patient health information.
""")

