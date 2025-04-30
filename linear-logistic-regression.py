# tính năng tải mô hình đã train
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import seaborn as sns
import joblib  # Thư viện để lưu và tải model

# --- Load Datasets ---
@st.cache_data
def load_advertising_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def load_social_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/social_network_ads.csv"
    df = pd.read_csv(url)
    return df

# --- Sidebar Navigation ---
st.sidebar.header("🧭 Navigation")
task = st.sidebar.radio("Choose Dataset", [
    "Advertising (Linear Regression)",
    "Social Network (Logistic Regression)"
])

# --- App Title ---
st.title("📈 Data Analysis & Machine Learning Models")

# --- Tabs ---
tabs = st.tabs(["Data Overview", "EDA", "Linear Regression", "Logistic Regression"])

# --- Logic: Linear Regression ---
if task == "Advertising (Linear Regression)":
    df = load_advertising_data()

    with tabs[0]:
        st.subheader("📄 Dataset Overview")
        st.write(df.head())

    with tabs[1]:
        st.subheader("📊 TV vs Sales")
        fig, ax = plt.subplots()
        ax.scatter(df["TV"], df["Sales"], alpha=0.7)
        ax.set_xlabel("TV Advertising Budget ($)")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with tabs[2]:
        st.subheader("🎯 Linear Regression Model")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        lr = st.number_input("Learning Rate (for display)", value=0.01)
        epochs = st.number_input("Epochs (for simulated loss)", value=200)

        if st.button("🚀 Train Linear Model"):
            X = df[["TV"]]
            y = df["Sales"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Fake loss curve
            losses = [mean_squared_error(y_train, model.predict(X_train)) / (i + 1) for i in range(epochs)]

            # Plot: Loss
            st.markdown("#### Training Progress")
            fig1, ax1 = plt.subplots()
            ax1.plot(range(epochs), losses, color="red")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            st.pyplot(fig1)

            # Plot: Predictions
            st.markdown("#### Model Predictions")
            fig2, ax2 = plt.subplots()
            ax2.scatter(X_train, y_train, label="Training Data", alpha=0.6)
            ax2.scatter(X_test, y_test, label="Test Data", alpha=0.6)
            ax2.plot(X, model.predict(X), color="red", label="Regression Line")
            ax2.set_xlabel("TV Advertising Budget ($)")
            ax2.set_ylabel("Sales ($)")
            ax2.legend()
            st.pyplot(fig2)

            # --- Save Model ---
            st.markdown("### Save Model")
            model_filename = "linear_regression_model.pkl"
            joblib.dump(model, model_filename)

            st.download_button(
                label="Download Model (Linear Regression)",
                data=open(model_filename, "rb").read(),
                file_name=model_filename,
                mime="application/octet-stream"
            )

# --- Logic: Logistic Regression ---
elif task == "Social Network (Logistic Regression)":
    df = load_social_data()
    df.rename(columns=lambda x: x.strip(), inplace=True)  # Clean column names

    with tabs[0]:
        st.subheader("📄 Dataset Overview")
        st.write(df.head())

    with tabs[1]:
        st.subheader("📊 Age vs Estimated Salary")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df["Age"], df["EstimatedSalary"], c=df["Purchased"], cmap="coolwarm", alpha=0.7)
        ax.set_xlabel("Age")
        ax.set_ylabel("Estimated Salary")
        legend1 = ax.legend(*scatter.legend_elements(), title="Purchased")
        ax.add_artist(legend1)
        st.pyplot(fig)

    with tabs[3]:
        st.subheader("🧠 Logistic Regression Model")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        C = st.number_input("Inverse of Regularization Strength (C)", value=1.0)

        if st.button("🚀 Train Logistic Model"):
            X = df[["Age", "EstimatedSalary"]]
            y = df["Purchased"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LogisticRegression(C=C, solver='lbfgs', max_iter=200)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.write(f"✅ Accuracy: **{acc:.2f}**")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            st.markdown("#### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            # --- Save Model ---
            st.markdown("### Save Model")
            model_filename = "logistic_regression_model.pkl"
            joblib.dump(model, model_filename)

            st.download_button(
                label="Download Model (Logistic Regression)",
                data=open(model_filename, "rb").read(),
                file_name=model_filename,
                mime="application/octet-stream"
            )