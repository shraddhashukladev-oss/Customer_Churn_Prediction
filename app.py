# app.py  — End-to-End EDA + ML + Prediction (Telco Churn)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Customer Churn — EDA + ML", layout="wide")
st.markdown("""
<style>
    .metric-card {background:#fff;border-radius:14px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,0.08)}
    .section {background:#f7f9fb;border-radius:14px;padding:16px}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Clean: TotalCharges may contain blanks -> make numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Keep rows even if TotalCharges is NaN (for EDA), but drop later for training
    return df

@st.cache_resource
def train_pipeline(df: pd.DataFrame):
    df_tr = df.copy()
    # Drop rows without target or essential fields
    df_tr = df_tr.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Target
    y = df_tr["Churn"].map({"Yes":1, "No":0})
    # Features: drop id + target
    X = df_tr.drop(columns=["customerID", "Churn"], errors="ignore")

    # Identify numeric/categorical
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [c for c in X.columns if c not in num_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop"
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cls = classification_report(y_test, y_pred, output_dict=True)

    # Persist meta to help build predict form with same categories
    meta = {
        "num_features": num_features,
        "cat_features": cat_features,
        "categories_map": {c: sorted(df_tr[c].dropna().unique().tolist()) for c in cat_features}
    }

    return pipe, {"acc": acc, "auc": auc, "cm": cm, "cls": cls}, meta

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    return fig

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve")
    ax.legend()
    return fig

# ---------------------------
# Load Data
# ---------------------------
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = load_data(DATA_PATH)

st.title("📊 Customer Churn — EDA + Machine Learning")

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview / EDA", "Train & Metrics", "Predict"])

# ---------------------------
# Tab 1: Overview / EDA
# ---------------------------
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Customers", df.shape[0])
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Features", df.shape[1])
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        churned = (df["Churn"]=="Yes").sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Churned Customers", churned)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        rate = churned / len(df) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Churn Rate", f"{rate:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Preview")
    st.dataframe(df.head(20))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Missing Values")
    st.write(df.isnull().sum())
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Churn", palette="pastel", ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Numerical Distributions")
    num_cols = ["tenure","MonthlyCharges","TotalCharges"]
    col = st.selectbox("Pick a numeric column", num_cols, index=0)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax2)
    ax2.set_title(f"Distribution — {col}")
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Categorical vs Churn")
    cat_cols = ["gender","SeniorCitizen","Partner","Dependents","PhoneService",
                "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                "Contract","PaperlessBilling","PaymentMethod"]
    ccol = st.selectbox("Pick a categorical column", cat_cols, index=14)
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x=ccol, hue="Churn", ax=ax3)
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
    ax3.set_title(f"{ccol} vs Churn")
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Correlation (Numerical)")
    fig4, ax4 = plt.subplots()
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Tab 2: Train & Metrics
# ---------------------------
with tab2:
    st.subheader("Model Training & Evaluation")
    with st.spinner("Training Logistic Regression pipeline..."):
        pipe, metrics, meta = train_pipeline(df)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{metrics['acc']*100:.2f}%")
    with c2:
        st.metric("ROC-AUC", f"{metrics['auc']:.3f}")

    st.markdown("#### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(metrics["cm"]))

    st.markdown("#### Classification Report")
    rep = pd.DataFrame(metrics["cls"]).T
    st.dataframe(rep.style.format(precision=3))

    st.info("The trained pipeline (preprocessing + model) is cached. Prediction tab uses the same pipeline to avoid feature mismatch.")

# ---------------------------
# Tab 3: Predict
# ---------------------------
with tab3:
    st.subheader("Predict Churn for a Single Customer")

    # Ensure we use the same pipeline & meta
    pipe, _, meta = train_pipeline(df)

    # Build a form that matches training features & categories
    with st.form("predict_form"):
        sc1, sc2, sc3 = st.columns(3)

        # Numeric fields
        with sc1:
            tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12)
        with sc2:
            monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)
        with sc3:
            total = st.number_input("TotalCharges", min_value=0.0, value=800.0)

        # Categorical fields (options from training data)
        inputs = {}
        for i, feat in enumerate(meta["cat_features"]):
            options = meta["categories_map"].get(feat, [])
            # sensible defaults
            default_idx = 0 if len(options)==0 else 0
            # 3 columns grid
            if i % 3 == 0:
                c1, c2, c3 = st.columns(3)
            col = [c1, c2, c3][i % 3]
            with col:
                if len(options) > 0:
                    val = st.selectbox(feat, options, index=default_idx, key=f"sel_{feat}")
                else:
                    val = st.text_input(feat, "")
                inputs[feat] = val

        # Assemble one-row DataFrame respecting training features
        data_dict = {**inputs,
                     "tenure": tenure,
                     "MonthlyCharges": monthly,
                     "TotalCharges": total}
        X_one = pd.DataFrame([data_dict])

        submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                prob = pipe.predict_proba(X_one)[0,1]
                pred = int(prob >= 0.5)
                if pred == 1:
                    st.error(f"⚠️ Likely to **Churn** (prob={prob:.2f})")
                else:
                    st.success(f"✅ Likely to **Stay** (prob={prob:.2f})")
            except Exception as e:
                st.exception(e)
                st.warning("If you see a feature-mismatch error, make sure your CSV is the standard Telco churn file.")


