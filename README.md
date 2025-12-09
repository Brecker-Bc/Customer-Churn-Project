# Telco Customer Churn Prediction

End to end churn prediction project using

- PostgreSQL as the data store  
- Python for data loading and model training  
- Scikit learn for the churn model  
- Streamlit for an interactive churn dashboard  

## Project structure

- `data/` - raw Telco churn CSV  
- `sql/` - schema definitions for PostgreSQL tables  
- `src/load_data_to_sql.py` - loads the CSV into the `customers` table  
- `src/train_model.py` - trains a Random Forest model and writes predictions to `churn_predictions`  
- `src/dashboard/streamlit_app.py` - Streamlit app for exploring churn risk  
- `src/utils/db.py` - shared database connection helper  

## Setup

1. Create a PostgreSQL database named `churn_db` on port 5433.  
2. Run `python src/load_data_to_sql.py` to create and populate the `customers` table.  
3. Run `python src/train_model.py` to train the model and write predictions into `churn_predictions`.  
4. Start the Streamlit dashboard:

   ```bash
   streamlit run src/dashboard/streamlit_app.py
