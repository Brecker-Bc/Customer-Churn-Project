from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg2://tylerbrecker@localhost:5433/churn_db"


def get_engine():
    """Return a SQLAlchemy engine for the churn database."""
    return create_engine(DB_URL)
