import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
COLORS = ["#4C9BE8", "#E8593C"]

@st.cache_data
def load_and_train(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "report": classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
        "cm": confusion_matrix(y_test, y_pred),
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1],
        "importances": pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False),
        "model": rf,
        "feature_cols": X.columns.tolist(),
    }
    return metrics


def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf)
    plt.close(fig)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telco Churn")
    st.markdown("Upload the **Telco Customer Churn** CSV to get started.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.caption("Model: Random Forest · 100 trees · max_depth=10")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Analysis")
st.markdown("End-to-end churn prediction pipeline · EDA → Model → Evaluation")

if uploaded is None:
    st.info("👈 Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the sidebar to begin.")
    st.markdown(
        """
        **Dataset columns required:** `customerID`, `gender`, `tenure`, `MonthlyCharges`,
        `TotalCharges`, `Contract`, `Churn`, and the rest of the standard Telco Churn features.

        You can download the dataset from
        [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
        """
    )
    st.stop()

df_raw = pd.read_csv(uploaded)

with st.spinner("Training Random Forest model…"):
    metrics = load_and_train(df_raw)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🏋️ Model", "📈 Evaluation", "🔮 Predict"])

# ── Tab 1: EDA ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Churn Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        df_raw["Churn"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", colors=COLORS, startangle=90, ax=ax
        )
        ax.set_title("Customer Churn Distribution")
        ax.set_ylabel("")
        fig_to_st(fig)

    with col2:
        st.markdown("**Churn by Contract Type**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df_raw, x="Contract", hue="Churn", palette=COLORS, ax=ax)
        ax.set_title("Churn Rate by Contract Type")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Number of Customers")
        fig_to_st(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Tenure Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        df_raw[df_raw["Churn"] == "Yes"]["tenure"].hist(
            bins=30, alpha=0.7, color=COLORS[1], label="Churned", ax=ax
        )
        df_raw[df_raw["Churn"] == "No"]["tenure"].hist(
            bins=30, alpha=0.7, color=COLORS[0], label="Retained", ax=ax
        )
        ax.set_title("Tenure Distribution: Churned vs Retained")
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Number of Customers")
        ax.legend()
        fig_to_st(fig)

    with col4:
        st.markdown("**Monthly Charges vs Churn**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df_raw, x="Churn", y="MonthlyCharges", palette=COLORS, ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Monthly Charges ($)")
        fig_to_st(fig)

    st.markdown("**Dataset Preview**")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# ── Tab 2: Model ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Random Forest Model")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']}%")
    c2.metric("ROC-AUC Score", f"{metrics['roc_auc']}")

    st.markdown("**Classification Report**")
    st.code(metrics["report"], language="text")

    st.markdown("**Top 10 Features Driving Churn**")
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics["importances"][:10].plot(kind="bar", color=COLORS[0], edgecolor="white", ax=ax)
    ax.set_title("Top 10 Features Driving Customer Churn")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    fig_to_st(fig)

# ── Tab 3: Evaluation ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            metrics["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax
        )
        ax.set_title("Confusion Matrix — Random Forest")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig_to_st(fig)

    with col2:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            metrics["fpr"], metrics["tpr"], color=COLORS[0], lw=2,
            label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})"
        )
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.set_title("ROC Curve — Random Forest Churn Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.legend(loc="lower right")
        fig_to_st(fig)

# ── Tab 4: Predict ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Predict Churn for a New Customer")
    st.markdown("Fill in the customer details below and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col3:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    col4, col5 = st.columns(2)
    with col4:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col5:
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                        float(tenure * monthly_charges), step=1.0)

    if st.button("🔮 Predict Churn", use_container_width=True):
        # Build a one-row raw dataframe matching training schema
        row = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges, "Churn": "No"  # placeholder
        }

        # Preprocess same way as training (use a union of train + this row)
        combined = pd.concat([df_raw.drop(columns=["customerID"]), pd.DataFrame([row])], ignore_index=True)
        combined["TotalCharges"] = pd.to_numeric(combined["TotalCharges"], errors="coerce")
        combined["TotalCharges"].fillna(combined["TotalCharges"].median(), inplace=True)

        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
        le = LabelEncoder()
        for col in binary_cols:
            combined[col] = le.fit_transform(combined[col])

        combined = pd.get_dummies(combined, drop_first=True)

        # Align columns with training
        feature_cols = metrics["feature_cols"]
        for c in feature_cols:
            if c not in combined.columns:
                combined[c] = 0
        combined = combined[feature_cols]

        new_customer = combined.iloc[[-1]]
        prob = metrics["model"].predict_proba(new_customer)[0][1]
        pred = "🔴 Likely to Churn" if prob >= 0.5 else "🟢 Likely to Stay"

        st.markdown("---")
        col_a, col_b = st.columns(2)
        col_a.metric("Churn Prediction", pred)
        col_b.metric("Churn Probability", f"{prob:.1%}")

        # Gauge-style bar
        st.progress(float(prob))
        if prob >= 0.7:
            st.error("⚠️ High churn risk — consider a retention offer.")
        elif prob >= 0.5:
            st.warning("⚠️ Moderate churn risk — monitor this customer.")
        else:
            st.success("✅ Low churn risk — customer appears satisfied.")
