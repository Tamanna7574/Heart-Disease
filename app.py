# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

DATA_FILE = "heart_dataset.csv"
MODEL_FILE = "heart_pipeline.pkl"

# possible target column names in different datasets
POSSIBLE_TARGETS = ["num", "target", "output", "condition", "heart_disease"]

numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']


@st.cache_resource
def load_or_train_model():
    # load dataset
    df = pd.read_csv(DATA_FILE)

    # auto-detect target column
    target_col = None
    for col in POSSIBLE_TARGETS:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error("❌ Target column not found in CSV. Expected one of: " + ", ".join(POSSIBLE_TARGETS))
        st.stop()

    st.info(f"🎯 Target column detected automatically: `{target_col}`")

    if os.path.exists(MODEL_FILE):
        st.success("📌 Loaded trained model")
        return joblib.load(MODEL_FILE), target_col

    st.warning("⚠️ Model not found — training new model...")

    X = df.drop(target_col, axis=1)
    y = (df[target_col] > 0).astype(int)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)
    st.success("🎉 Model trained & saved successfully")

    return pipeline, target_col


pipeline, target_col = load_or_train_model()
st.title("❤️ Heart Disease Prediction Dashboard")

menu = st.sidebar.radio("📌 Navigate", ["Home", "Prediction", "Analysis", "Contact"])

if menu == "Home":
    st.write("Welcome to the ML Heart Disease Prediction App")

elif menu == "Prediction":
    st.header("🩺 Patient Details")

    user = {
        "age": st.number_input("Age", 1, 120, 50),
        "trestbps": st.number_input("Resting BP", 80, 200, 120),
        "chol": st.number_input("Cholesterol", 100, 600, 240),
        "thalch": st.number_input("Max Heart Rate", 60, 250, 150),
        "oldpeak": st.number_input("ST Depression", 0.0, 10.0, 1.0),
        "ca": st.number_input("Major Vessels (0-4)", 0, 4, 0),
        "sex": st.selectbox("Sex", ["0", "1"]),
        "cp": st.selectbox("Chest Pain Type", ["0", "1", "2", "3"]),
        "fbs": st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0", "1"]),
        "restecg": st.selectbox("Resting ECG", ["0", "1", "2"]),
        "exang": st.selectbox("Exercise Induced Angina", ["0", "1"]),
        "slope": st.selectbox("Slope", ["0", "1", "2"]),
        "thal": st.selectbox("Thalassemia", ["0", "1", "2"])
    }

    df_user = pd.DataFrame([user])

    if st.button("🔍 Predict"):
        result = pipeline.predict(df_user)[0]
        if result == 1:
            st.error("⚠️ High Risk — Patient may have heart disease")
        else:
            st.success("✅ Low Risk — Patient is likely healthy")

elif menu == "Analysis":
    st.header("📊 Dataset Analysis")
    df = pd.read_csv(DATA_FILE)
    st.dataframe(df.head())

elif menu == "Contact":
    st.write("Made with ❤️ by Tamanna")
