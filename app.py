"""
Student Burnout & Dropout Risk Prediction System
Streamlit Web Application - UI/UX Enhanced
"""

import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Student Burnout Prediction",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - Modern theme, soft palette, card styling
# ============================================================
CUSTOM_CSS = """
<style>
/* Overall app background - soft off-white */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
}

/* Main content area padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* Section headers with background highlight */
.section-header {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: white !important;
    padding: 0.75rem 1.25rem;
    border-radius: 12px;
    margin: 1.5rem 0 1rem 0;
    font-weight: 600;
    font-size: 1.15rem;
}


/* Slider styling - thicker track, consistent color */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
}
.stSlider > div > div > div > div {
    background: #6366f1 !important;
}

/* Predict button - full width, gradient, rounded */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.85rem 1.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    transform: translateY(-1px);
}

/* Result card - High Risk (red/coral) */
.result-card-high {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-left: 5px solid #ef4444;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
}

/* Result card - Low Risk (green) */
.result-card-low {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left: 5px solid #22c55e;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15);
}

/* Risk label - large font */
.risk-label {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

/* Probability display */
.prob-display {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin: 0.75rem 0 !important;
}

/* Interventions section - soft shaded box */
.interventions-box {
    background: rgba(248, 250, 252, 0.9);
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    margin-top: 1rem;
}

/* Title centering */
.main h1 {
    text-align: center;
    margin-bottom: 0.5rem;
}

/* Subtitle styling */
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
}
</style>
"""

# ============================================================
# SENTIMENT MAPPING
# ============================================================
SENTIMENT_MAP = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

# ============================================================
# MODEL & SCALER PATHS
# ============================================================
MODEL_PATH = "burnout_model.pkl"
SCALER_PATH = "scaler.pkl"

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================


def load_model_and_scaler():
    """Load the trained Random Forest model and StandardScaler from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}. Please add the file to the project directory.")
        return None, None
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found: {SCALER_PATH}. Please add the file to the project directory.")
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # ----- Apply custom CSS theme -----
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ----- 1. Page Header (centered, with icons) -----
    st.title("🎓 Student Burnout & Dropout Risk Prediction System")
    st.markdown('<p class="subtitle">Enter student behavioural and academic details to predict burnout risk.</p>', unsafe_allow_html=True)

    # ----- 2. Input Section -----
    st.markdown('<p class="section-header">📊 Input Parameters</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider(
            "🎓 Attendance (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1,
            help="How regularly the student attends classes"
        )
        lms_logins = st.slider(
            "💻 LMS Logins per Week",
            min_value=0,
            max_value=20,
            value=8,
            step=1,
            help="Measures student engagement with the learning platform"
        )
        lateness = st.slider(
            "⏰ Assignment Lateness (Days)",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Indicates academic stress or burnout"
        )

    with col2:
        feedback_sentiment = st.selectbox(
            "😊 Feedback Sentiment",
            options=["Positive", "Neutral", "Negative"],
            index=0,
            help="Represents emotional state of the student"
        )
        activity_irregularity = st.slider(
            "📉 Activity Irregularity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            format="%.1f",
            help="Shows sudden behaviour changes (important burnout signal)"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state for persisting prediction results
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    # ----- 3. Predict Button (full width, gradient) -----
    if st.button("🔍 Predict Burnout Risk", type="primary", use_container_width=True):
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            st.stop()

        feedback_sentiment_score = SENTIMENT_MAP[feedback_sentiment]
        irregularity_flag = 1 if float(activity_irregularity) > 0.7 else 0
        assignment_lateness = float(lateness)
        activity_irregularity_score = float(activity_irregularity)

        try:
            scaler_df = pd.DataFrame([{
                "LMS_Logins_per_Week": float(lms_logins),
                "Assignment_Lateness_Days": assignment_lateness,
                "Feedback_Sentiment_Score": float(feedback_sentiment_score),
                "Activity_Irregularity_Score": activity_irregularity_score,
            }])
            scaled_4 = scaler.transform(scaler_df)[0]

            model_input = pd.DataFrame([{
                "Attendance": float(attendance),
                "Avg_Weekly_Logins": scaled_4[0],
                "Assignment_Lateness_Days": scaled_4[1],
                "Avg_Feedback_Sentiment": scaled_4[2],
                "Activity_Irregularity_Score": scaled_4[3],
                "Irregularity_Flag": int(irregularity_flag),
            }])

            prediction = model.predict(model_input)[0]
            proba = model.predict_proba(model_input)[0]
            classes = model.classes_
            high_risk_idx = 1 if len(classes) == 2 else 0
            for i, c in enumerate(classes):
                if "high" in str(c).lower() or c == 1:
                    high_risk_idx = i
                    break
            high_risk_prob = float(proba[high_risk_idx])

            is_high_risk = (
                str(prediction).lower() in ["high risk", "high", "1"] or prediction == 1
            )
            prob_percent = (high_risk_prob if is_high_risk else (1 - high_risk_prob)) * 100
            st.session_state.prediction_result = {
                "is_high_risk": is_high_risk,
                "prob_percent": prob_percent,
            }

        except Exception as e:
            st.session_state.prediction_result = {"error": str(e)}

    # ----- 4. Result Display (colored cards) -----
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        if "error" in result:
            st.error(f"Prediction error: {result['error']}")
            st.info("Scaler expects 4 features; model expects 6. Check that your training pipeline matches.")
        else:
            st.markdown('<p class="section-header">📊 Prediction Result</p>', unsafe_allow_html=True)
            prob_percent = result["prob_percent"]
            is_high_risk = result["is_high_risk"]

            if is_high_risk:
                st.markdown(
                    f"""
                    <div class="result-card-high">
                        <div class="risk-label">⚠️ High Burnout Risk</div>
                        <div class="prob-display">Probability: {prob_percent:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown('<p class="section-header">Recommended Interventions</p>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="interventions-box">
                        <p>🧠 <strong>Academic counseling</strong> – Schedule one-on-one sessions</p>
                        <p>👩‍🏫 <strong>Faculty support</strong> – Increased check-ins and progress tracking</p>
                        <p>💬 <strong>Mental health guidance</strong> – Connect with campus counseling services</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-low">
                        <div class="risk-label">✅ Low Burnout Risk</div>
                        <div class="prob-display">Probability: {prob_percent:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown('<p class="section-header">Recommended Interventions</p>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="interventions-box">
                        <p>💪 <strong>Motivation messages</strong> – Continue positive reinforcement</p>
                        <p>📊 <strong>Regular academic monitoring</strong> – Maintain routine progress checks</p>
                        <p>🎉 <strong>Positive reinforcement</strong> – Encourage continued engagement</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
