import pandas as pd
from sqlalchemy import create_engine

import os

# Database configuration
DB_USER     = os.getenv('DB_USER', 'user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST     = os.getenv('DB_HOST', 'postgres')
DB_PORT     = os.getenv('DB_PORT', '5432')
DB_NAME     = os.getenv('DB_NAME', 'churn_db')

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def load_data():
    """Load customer data from PostgreSQL."""
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, engine)
    return df
