"""
Database Connection Module

Handles the PostgreSQL database connections for the application.
Ensure .env file is configured with DB credentials.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    # TODO: Load these from environment variables later
    conn = psycopg2.connect(
        host="localhost",
        database="hybrid_search",
        user="admin",
        password="password",
        port="5432"
    )
    return conn

if __name__ == "__main__":
    # Quick sanity check
    try:
        connection = get_db_connection()
        print("Successfully connected to the database.")
        connection.close()
    except Exception as e:
        print(f"Database connection failed: {e}")   