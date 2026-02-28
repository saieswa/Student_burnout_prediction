# 🎓 Student Burnout & Dropout Risk Prediction System

A Streamlit web application that predicts student burnout and dropout risk using a trained Random Forest machine learning model.

## Features

- **Input Parameters**: Attendance, LMS logins, assignment lateness, feedback sentiment, activity irregularity
- **ML Prediction**: Trained Random Forest classifier with StandardScaler
- **Interventions**: Actionable recommendations for high/low risk students
- **Modern UI**: Clean, professional design with gradient theme

## Setup

```bash
pip install streamlit joblib numpy pandas scikit-learn
```

## Run

```bash
streamlit run app.py
```

## Project Structure

```
student-burnout-app/
├── app.py           # Streamlit application
├── burnout_model.pkl # Trained Random Forest model
├── scaler.pkl       # StandardScaler for input normalization
└── README.md
```

## Author

saieswa
