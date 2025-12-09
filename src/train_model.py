import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from utils.db import get_engine


def main():
    engine = get_engine()

    # read data
    df = pd.read_sql("SELECT * FROM customers", engine)

    # label
    df["churn_label"] = (df["churn"] == "Yes").astype(int)

    customer_ids = df["customer_id"]
    X = df.drop(columns=["customer_id", "churn", "churn_label"])
    y = df["churn_label"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
        X, y, customer_ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    y_proba_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    print("Test AUC:", roc_auc_score(y_test, y_proba_test))
    print(classification_report(y_test, y_pred_test))

    # predict for all customers
    all_proba = clf.predict_proba(X)[:, 1]

    predictions_df = pd.DataFrame({
        "customer_id": customer_ids,
        "prediction_date": datetime.date.today(),
        "churn_probability": all_proba,
        "model_version": "rf_v1",
    })

    # ensure predictions table exists
    with engine.begin() as conn:
        conn.execute(text(open("sql/create_churn_predictions.sql").read()))

        conn.execute(
            text("DELETE FROM churn_predictions WHERE prediction_date = :d"),
            {"d": datetime.date.today()},
        )

    predictions_df.to_sql(
        "churn_predictions",
        engine,
        if_exists="append",
        index=False,
    )

    print(f"Wrote predictions for {len(predictions_df)} customers.")


if __name__ == "__main__":
    main()
