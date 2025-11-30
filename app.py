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
TARGET_COL = "num"

numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']


@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        st.info("📌 Loaded trained model")
        return joblib.load(MODEL_FILE)

    st.warning("⚠️ Model not found — training new model...")
    df = pd.read_csv(DATA_FILE)

    X = df.drop(TARGET_COL, axis=1)
    y = (df[TARGET_COL] > 0).astype(int)

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
    return pipeline


pipeline = load_or_train_model()


# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------

st.title("❤️ Heart Disease Prediction Dashboard")

menu = st.sidebar.radio("📌 Navigate", ["Home", "Prediction", "Analysis", "Contact"])

# --------------------------------------------------------------
# Home
# --------------------------------------------------------------
if menu == "Home":
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=200)
    st.write("""
    Welcome to the **Heart Disease Prediction System**  
    Built using **Python + Machine Learning + Streamlit**  
    Navigate using the sidebar to test prediction & explore data insights.
    """)

# --------------------------------------------------------------
# Prediction
# --------------------------------------------------------------
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

# --------------------------------------------------------------
# Analysis
# --------------------------------------------------------------
elif menu == "Analysis":
    st.header("📊 Dataset Analysis")

    df = pd.read_csv(DATA_FILE)
    st.subheader("🔹 Preview Dataset")
    st.dataframe(df.head())

    st.subheader("📈 Numeric Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)

    st.subheader("🗂️ Categorical Feature Distributions")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, col in enumerate(categorical_cols):
        sns.countplot(x=col, data=df, ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)

# --------------------------------------------------------------
# Contact
# --------------------------------------------------------------
elif menu == "Contact":
    st.header("📞 Contact")
    st.write("""
    📧 Email: tamanna@example.com  
    🔗 GitHub: https://github.com  
    💼 Developer: Tamanna | CSE Project  
    """)

st.markdown(
    "<div style='text-align:center;margin-top:20px;'>Made with ❤️ by Tamanna</div>",
    unsafe_allow_html=True
)
