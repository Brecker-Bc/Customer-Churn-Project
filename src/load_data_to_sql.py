import pandas as pd
from sqlalchemy import text

from utils.db import get_engine


def main():
    df = pd.read_csv("data/TelcoChurn.csv")

    # standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # fix total charges
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

    # rename to match SQL schema
    df = df.rename(columns={
        "customerid": "customer_id",
        "seniorcitizen": "senior_citizen",
        "phoneservice": "phone_service",
        "multiplelines": "multiple_lines",
        "internetservice": "internet_service",
        "onlinesecurity": "online_security",
        "onlinebackup": "online_backup",
        "deviceprotection": "device_protection",
        "techsupport": "tech_support",
        "streamingtv": "streaming_tv",
        "streamingmovies": "streaming_movies",
        "paperlessbilling": "paperless_billing",
        "paymentmethod": "payment_method",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
    })

    engine = get_engine()

    create_customers_table = open("sql/create_customers.sql").read()

    with engine.begin() as conn:
        conn.execute(text(create_customers_table))

    df.to_sql("customers", engine, if_exists="append", index=False)

    print(f"Loaded {len(df)} customers into SQL.")


if __name__ == "__main__":
    main()
