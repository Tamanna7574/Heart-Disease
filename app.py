# -*- coding: utf-8 -*-
"""
Heart Disease Prediction App
Cleaned version for deployment (Streamlit Cloud / local Python)
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load dataset and model
# -----------------------------
try:
    pipeline = joblib.load("heart_pipeline.pkl")
except:
    st.warning("Pipeline not found. Training a new model...")
    df = pd.read_csv("C:\Users\HP\Desktop\heart_dataset.csv")

    numeric_cols = ['age','trestbps','chol','thalch','oldpeak','ca']
    categorical_cols = ['sex','cp','fbs','restecg','exang','slope','thal']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X = df.drop('num', axis=1)
    y = (df['num'] > 0).astype(int)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "heart_pipeline.pkl")
    st.success("✅ Trained and saved pipeline as heart_pipeline.pkl")

# -----------------------------
# Streamlit app layout
# -----------------------------

st.title("❤️ Heart Disease Prediction Dashboard")

tab = st.sidebar.radio("📌 Navigate", ["🏠 Home", "🩺 Prediction", "📊 Analysis", "📞 Contact"])

# -----------------------------
# Home
# -----------------------------
if tab == "🏠 Home":
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=250)
    st.write("""
        Welcome! This interactive app predicts heart disease risk using **Machine Learning**.
        - 🩺 **Prediction Tool**
        - 📊 **Analysis Dashboard**
        - 📈 **Visual Reports**
    """)
    st.success("🔬 Developed with Python + Streamlit")

# -----------------------------
# Prediction
# -----------------------------
elif tab == "🩺 Prediction":
    st.header("🩺 Enter Patient Details")

    user_input = {
        "age": st.number_input("Age", min_value=1, max_value=120, value=46),
        "trestbps": st.number_input("Resting Blood Pressure", value=120),
        "chol": st.number_input("Cholesterol", value=200),
        "thalch": st.number_input("Max Heart Rate Achieved", value=150),
        "oldpeak": st.number_input("ST Depression", value=1.0),
        "ca": st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0),
        "sex": st.selectbox("Sex", ["Male", "Female"]),
        "cp": st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
        "fbs": st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
        "restecg": st.selectbox("Resting ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"]),
        "exang": st.selectbox("Exercise Induced Angina", [0, 1]),
        "slope": st.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"]),
        "thal": st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])
    }

    user_input_df = pd.DataFrame([user_input])
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for col in categorical_cols:
        user_input_df[col] = user_input_df[col].astype(str).str.lower()

    if st.button("🔍 Predict Risk"):
        prediction = pipeline.predict(user_input_df)[0]
        if prediction == 1:
            st.error("⚠️ High risk of heart disease")
        else:
            st.success("✅ Low risk of heart disease")

# -----------------------------
# Analysis
# -----------------------------
elif tab == "📊 Analysis":
    st.header("📊 Dataset Analysis")
    df = pd.read_csv("heart_dataset.csv")
    numeric_cols = ['age','trestbps','chol','thalch','oldpeak','ca']
    categorical_cols = ['sex','cp','fbs','restecg','exang','slope','thal']

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("📈 Numeric Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🗂️ Categorical Feature Distributions")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            sns.countplot(x=col, data=df, ax=axes[i], palette='pastel')
            axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# Contact
# -----------------------------
elif tab == "📞 Contact":
    st.header("📞 Contact Us")
    st.write("""
        💡 Questions or feedback? Reach out!
        - 📧 Email: **tamanna@example.com**
        - 🔗 LinkedIn: [Tamanna](https://linkedin.com)
        - 🏫 Project: CSE Final Year
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown('<div style="text-align:center;margin-top:20px;">Made with ❤️ by Tamanna | CSE Project</div>', unsafe_allow_html=True)
