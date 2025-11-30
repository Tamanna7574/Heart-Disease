# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

DATA_FILE = "heart_dataset.csv"
MODEL_FILE = "heart_pipeline.pkl"
TARGET_COL = "target"

# ------------------------------
# Load or Train Model
# ------------------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    st.warning("🔁 Model not found — training a new model...")

    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

    # Separate features & target
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Identify numeric and categorical features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)

    st.success("✅ Model trained & saved!")
    return pipeline

pipeline = load_or_train_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("❤️ Heart Disease Prediction Dashboard")

menu = st.sidebar.radio("📌 Menu", ["🏠 Home", "🩺 Prediction", "📊 Data Analysis"])

# ------------------------------
# Home
# ------------------------------
if menu == "🏠 Home":
    st.write("""
    Welcome to the **Heart Disease Prediction Web App**  
    Built using **Machine Learning + Streamlit** ❤️  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=260)
    st.info("Navigate to **Prediction** to check patient's heart disease risk.")

# ------------------------------
# Prediction Page
# ------------------------------
elif menu == "🩺 Prediction":
    st.header("🧑‍⚕️ Patient Details")

    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')
    input_cols = df.drop(TARGET_COL, axis=1).columns.tolist()
    user_input = {}

    # -------------------
    # Collect user input
    # -------------------
    for col in input_cols:
        if df[col].dtype == 'object':
            user_input[col] = st.selectbox(col, df[col].unique())
        elif df[col].dtype == 'bool' or df[col].nunique() == 2:
            # Ensure valid 0/1 input for one-hot columns
            user_input[col] = int(st.selectbox(col, [0, 1]))
        else:
            user_input[col] = st.number_input(col, value=float(df[col].median()))

    user_input_df = pd.DataFrame([user_input])

    # -------------------
    # Align columns with training
    # -------------------
    try:
        # Make sure all expected columns exist
        for col in pipeline.named_steps['preprocessor'].get_feature_names_out():
            if col not in user_input_df.columns:
                user_input_df[col] = 0
        user_input_df = user_input_df[pipeline.named_steps['preprocessor'].get_feature_names_out()]
    except:
        pass  # fallback if feature_names_in_ not available

    # -------------------
    # Prediction
    # -------------------
    if st.button("🔍 Predict"):
        prediction = pipeline.predict(user_input_df)[0]
        if prediction == 1:
            st.error("⚠️ Patient is at **high risk of heart disease**")
        else:
            st.success("💚 Patient is at **low risk of heart disease**")

# ------------------------------
# Data Analysis
# ------------------------------
elif menu == "📊 Data Analysis":
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')
    st.header("📊 Dataset Overview")
    st.dataframe(df.head())

    st.subheader("📈 Numeric Feature Distributions")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove(TARGET_COL)

    fig, axes = plt.subplots(len(numeric_cols)//3 + 1, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("<br><center>Made with ❤️ by Tamanna | CSE Project</center>", unsafe_allow_html=True)
