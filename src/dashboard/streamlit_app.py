import os
import sys

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import text

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Make sure we can import utils.db (project root on sys.path)
# -------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.db import get_engine  # noqa: E402

sns.set(style="whitegrid")


@st.cache_data(ttl=300)
def load_latest_predictions():
    """Load latest predictions joined with customer attributes."""
    engine = get_engine()
    query = """
        WITH latest AS (
            SELECT MAX(prediction_date) AS d
            FROM churn_predictions
        )
        SELECT
            c.customer_id,
            c.gender,
            c.senior_citizen,
            c.partner,
            c.dependents,
            c.tenure,
            c.contract,
            c.internet_service,
            c.payment_method,
            c.monthly_charges,
            c.total_charges,
            c.churn AS actual_churn,
            p.prediction_date,
            p.churn_probability,
            p.model_version
        FROM customers c
        JOIN churn_predictions p
          ON c.customer_id = p.customer_id
        JOIN latest l
          ON p.prediction_date = l.d;
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    # numeric label for actual churn
    df["churn_label"] = (df["actual_churn"] == "Yes").astype(int)
    return df


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    st.pyplot(fig)


def main():
    st.set_page_config(
        page_title="Telco Churn Dashboard",
        layout="wide",
    )

    st.title("Telco Customer Churn Prediction")
    st.write(
        "SQL-backed churn model using Random Forest on the Telco dataset. "
        "Predictions are stored in PostgreSQL and visualized here."
    )

    df = load_latest_predictions()
    if df.empty:
        st.warning("No predictions found. Run train_model.py first.")
        return

    # Convenience vectors for model evaluation
    y_true = df["churn_label"].values
    y_proba = df["churn_probability"].values

    # -------------------------------------------------------------------
    # Tabs: Overview | EDA | Model performance | Customer explorer
    # -------------------------------------------------------------------
    tab_overview, tab_eda, tab_model, tab_customers = st.tabs(
        ["Overview", "EDA", "Model performance", "Customer explorer"]
    )

    # -------------------------------------------------------------------
    # OVERVIEW
    # -------------------------------------------------------------------
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Number of customers", len(df))

        with col2:
            st.metric("Observed churn rate", f"{df['churn_label'].mean():.3f}")

        with col3:
            st.metric(
                "Average predicted churn probability",
                f"{df['churn_probability'].mean():.3f}",
            )

        with col4:
            st.metric("Model version", df["model_version"].iloc[0])

        st.subheader("Predicted churn probability distribution")
        st.bar_chart(df["churn_probability"])

        st.subheader("Average predicted churn by contract type")
        st.bar_chart(
            df.groupby("contract")["churn_probability"].mean().sort_values()
        )

    # -------------------------------------------------------------------
    # EDA (similar spirit to the notebook)
    # -------------------------------------------------------------------
    with tab_eda:
        st.subheader("Churn rate by contract type")
        churn_by_contract = (
            df.groupby("contract")["churn_label"].mean().sort_values().reset_index()
        )
        st.bar_chart(
            churn_by_contract.set_index("contract")["churn_label"]
        )

        st.subheader("Churn rate by internet service")
        churn_by_internet = (
            df.groupby("internet_service")["churn_label"]
            .mean()
            .sort_values()
            .reset_index()
        )
        st.bar_chart(
            churn_by_internet.set_index("internet_service")["churn_label"]
        )

        st.subheader("Tenure distribution by churn status")
        col_left, col_right = st.columns(2)

    with col_left:
        st.write("Customers who stayed (churn = No)")
        stayed = df[df["churn_label"] == 0]["tenure"]

        fig, ax = plt.subplots()
        ax.hist(stayed, bins=20, color="tab:blue", alpha=0.8)
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Count")
        ax.set_title("Tenure distribution – stayed")
        st.pyplot(fig)

    with col_right:
        st.write("Customers who churned (churn = Yes)")
        churned = df[df["churn_label"] == 1]["tenure"]

        fig, ax = plt.subplots()
        ax.hist(churned, bins=20, color="tab:orange", alpha=0.8)
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Count")
        ax.set_title("Tenure distribution – churned")
        st.pyplot(fig)


        
        st.markdown(
            "_These views mirror the kind of churn-by-segment analysis from the "
            "original Telco churn notebook: contract, internet service, and tenure "
            "are all strongly related to churn behavior._"
        )

    # -------------------------------------------------------------------
    # MODEL PERFORMANCE
    # -------------------------------------------------------------------
    with tab_model:
        st.subheader("Model performance")

        threshold = st.slider(
            "Classification threshold (for Churn vs No churn)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

        y_pred = (y_proba >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_proba)
        st.metric("AUC (ROC)", f"{auc:.3f}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.write("Confusion matrix")
            plot_confusion_matrix(y_true, y_pred)

        with col_right:
            st.write("Classification report")
            report = classification_report(
                y_true, y_pred, target_names=["No churn", "Churn"]
            )
            st.code(report, language="text")

        st.markdown(
            "_You can adjust the threshold to trade off between catching more true "
            "churners (recall) and reducing false alarms (precision), just like in a "
            "real churn-retention scenario._"
        )

    # -------------------------------------------------------------------
    # CUSTOMER EXPLORER
    # -------------------------------------------------------------------
    with tab_customers:
        st.subheader("Customer explorer")

        contract_filter = st.multiselect(
            "Contract type",
            options=sorted(df["contract"].unique()),
            default=sorted(df["contract"].unique()),
        )

        internet_filter = st.multiselect(
            "Internet service",
            options=sorted(df["internet_service"].unique()),
            default=sorted(df["internet_service"].unique()),
        )

        prob_filter = st.slider(
            "Minimum predicted churn probability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

        filtered = df[
            (df["contract"].isin(contract_filter))
            & (df["internet_service"].isin(internet_filter))
            & (df["churn_probability"] >= prob_filter)
        ].copy()

        st.write(f"{len(filtered)} customers match filters")

        st.dataframe(
            filtered.sort_values("churn_probability", ascending=False)[
                [
                    "customer_id",
                    "actual_churn",
                    "contract",
                    "internet_service",
                    "tenure",
                    "monthly_charges",
                    "churn_probability",
                ]
            ]
        )


if __name__ == "__main__":
    main()
