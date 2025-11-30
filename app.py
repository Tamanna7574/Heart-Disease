# -*- coding: utf-8 -*-
"""
Heart Disease Prediction App (Final Deployment Ready)
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

DATA_FILE = "heart_dataset.csv"
MODEL_FILE = "heart_pipeline.pkl"
TARGET_COL = "target"   # <-- change this if your CSV uses CLASS/OUTPUT/NUM

# -----------------------------
# Load dataset and model
# -----------------------------
@st.cache_resource
def load_or_train_model():
    try:
        return joblib.load(MODEL_FILE)
    except:
        st.warning("⚠️ Model not found — training new model...")
        df = pd.read_csv(DATA_FILE)

        numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ])

        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_FILE)
        st.success("✔ Model trained & saved!")

        return pipeline

pipeline = load_or_train_model()

# -----------------------------
# Streamlit app UI
# -----------------------------
st.title("❤️ Heart Disease Prediction Dashboard")
tab = st.sidebar.radio("📌 Menu", ["🏠 Home", "🩺 Prediction", "📊 Analysis", "📞 Contact"])


# ----------------------------- HOME
if tab == "🏠 Home":
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=250)
    st.write("""
    Welcome! This app predicts **heart disease risk** using Machine Learning.
    - 🩺 Smart prediction
    - 📊 Visual data analytics
    - 🔥 Auto-trained ML model
    """)


# ----------------------------- PREDICTION
elif tab == "🩺 Prediction":
    st.header("🩺 Enter Patient Details")

    user_input = {
        "age": st.number_input("Age", min_value=1, max_value=120, value=46),
        "trestbps": st.number_input("Resting Blood Pressure", value=120),
        "chol": st.number_input("Cholesterol", value=200),
        "thalch": st.number_input("Max Heart Rate Achieved", value=150),
        "oldpeak": st.number_input("ST Depression", value=1.0),
        "ca": st.number_input("Major Vessels (0–4)", min_value=0, max_value=4, value=0),
        "sex": st.selectbox("Sex", ["male", "female"]),
        "cp": st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
        "fbs": st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1]),
        "restecg": st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"]),
        "exang": st.selectbox("Exercise Induced Angina", [0, 1]),
        "slope": st.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"]),
        "thal": st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])
    }

    user_df = pd.DataFrame([user_input])

    if st.button("🔍 Predict"):
        result = pipeline.predict(user_df)[0]
        if result == 1:
            st.error("⚠️ High risk of heart disease")
        else:
            st.success("✅ Low risk of heart disease")


# ----------------------------- ANALYSIS
elif tab == "📊 Analysis":
    st.header("📊 Dataset Analytics")
    df = pd.read_csv(DATA_FILE)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Numeric Distributions")
    numeric_cols = ['age','trestbps','chol','thalch','oldpeak','ca']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)

    st.subheader("🗂 Categorical Distributions")
    categorical_cols = ['sex','cp','fbs','restecg','exang','slope','thal']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, col in enumerate(categorical_cols):
        sns.countplot(x=col, data=df, ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)


# ----------------------------- CONTACT
elif tab == "📞 Contact":
    st.header("📞 Contact")
    st.write("""
    👉 Developer: **Tamanna**  
    📩 Email: tamanna@example.com  
    🔗 LinkedIn: https://linkedin.com  
    """)

st.markdown('<div style="text-align:center;margin-top:20px;">Made with ❤️ by Tamanna</div>', unsafe_allow_html=True)
