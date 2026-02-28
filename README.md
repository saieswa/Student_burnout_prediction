# 🎓 Student Burnout & Dropout Risk Prediction System

A **behavioural analytics–driven web application** that predicts **student burnout and dropout risk** using machine learning.  
The system combines **academic performance data** with **engineered behavioural indicators** to provide:

- ✅ A **Risk Score (0–100)**
- ✅ **Burnout Risk Classification** (Low / High)
- ✅ **Key behavioural triggers**
- ✅ **Recommended intervention strategies**

The application is implemented as an **interactive Streamlit web app** powered by a **Random Forest model**.

---

## 📌 Problem Statement

Educational institutions often identify student burnout and dropout risk **after academic performance has already declined**.  
However, **early behavioural signals**—such as reduced engagement, delayed submissions, emotional feedback, and irregular activity—can indicate burnout much earlier.

This project aims to **detect burnout risk early** using behavioural analytics and machine learning, enabling **timely and targeted interventions**.

---

## 🎯 Objectives

- Predict **student burnout risk** (Low / High)
- Generate a **risk score (0–100)** representing prediction confidence
- Identify **key behavioural triggers** influencing the prediction
- Recommend **practical intervention strategies**
- Provide an **interactive, user-friendly web interface**

---

## 📊 Dataset Description

### Dataset Type
**Public dataset with synthetically generated behavioural features**

### Dataset Source
- Kaggle – Student Performance / Dropout–related dataset  
(Add the exact Kaggle link used)

### Why This Dataset Fits Behavioural Analytics
- Contains **academic performance and attendance indicators**
- Can be extended to model **behavioural engagement patterns**
- Suitable for predicting **burnout and dropout risk**

---

## 🧠 Behavioural Feature Engineering

Since **real behavioural datasets (LMS logs, sentiment, irregularity)** are not publicly available due to privacy constraints, behavioural features were **simulated using rule-based logic**.

### Engineered Behavioural Features

| Feature Name | Description |
|-------------|------------|
| Attendance | Percentage of classes attended |
| Avg_Weekly_Logins | Simulated LMS engagement level |
| Assignment_Lateness_Days | Academic stress indicator |
| Avg_Feedback_Sentiment | Emotional wellbeing proxy |
| Activity_Irregularity_Score | Measures sudden changes in behaviour |
| Irregularity_Flag | Binary indicator for high irregularity (> 0.7) |

### Data Assumptions
- Low engagement + high irregularity → higher burnout risk
- Stable behaviour may indicate low burnout even with low attendance
- Behavioural **patterns** matter more than individual metrics

---

## ⚙️ Machine Learning Model

### Model Used
**Random Forest Classifier**

### Why Random Forest?
- Handles **non-linear behavioural interactions**
- Performs well on **tabular data**
- Provides **feature importance** for explainability
- Robust and suitable for hackathons

### Supporting Components
- **StandardScaler** for feature normalization
- Logistic Regression used as a **baseline model**

---

## 📈 Model Outputs

The system provides the following outputs:

### 1️⃣ Risk Score
- Probability score scaled between **0–100**

### 2️⃣ Burnout Risk Classification
- **Low Burnout Risk**
- **High Burnout Risk**

### 3️⃣ Key Behavioural Triggers
- Attendance trends
- LMS engagement levels
- Assignment lateness
- Feedback sentiment
- Behavioural irregularity

### 4️⃣ Recommended Intervention Strategy

#### ⚠️ High Burnout Risk
- Academic counseling  
- Faculty intervention  
- Mental health support  

#### ✅ Low Burnout Risk
- Motivation messages  
- Regular academic monitoring  
- Positive reinforcement  

---

## 🌐 Web Application (Streamlit)

### Features
- Interactive input sliders and dropdowns
- Real-time ML prediction
- Color-coded risk display
- Clear intervention recommendations
- Modern, professional UI

### Input Parameters
- Attendance (%)
- LMS Logins per Week
- Assignment Lateness (days)
- Feedback Sentiment
- Activity Irregularity Score

---

## 🚀 Run the App
```bash
pip install streamlit joblib numpy pandas scikit-learn
streamlit run app.py
