CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id     SERIAL PRIMARY KEY,
    customer_id       VARCHAR(50) REFERENCES customers(customer_id),
    prediction_date   DATE DEFAULT CURRENT_DATE,
    churn_probability NUMERIC(5,4),
    model_version     VARCHAR(50)
);
