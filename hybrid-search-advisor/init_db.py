import psycopg2
import random

def setup_database():
    print("Connecting to database on port 5454...")
    conn = psycopg2.connect(
        host="localhost", database="hybrid_search", user="admin", password="password", port="5454"
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print("Creating vector extension and table...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("""
        DROP TABLE IF EXISTS documents;
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            author VARCHAR(255),
            category VARCHAR(100),
            embedding VECTOR(3) 
        );
    """)

    print("Inserting 1,000 dummy documents...")
    # Creating an uneven distribution so the ML Router has something to calculate
    categories = ['Physics'] * 100 + ['Computer Science'] * 850 + ['Biology'] * 50
    authors = ['Alice'] * 990 + ['Srinivasa Ramanujan'] * 10 

    for i in range(1000):
        vec = f"[{random.uniform(0,1)}, {random.uniform(0,1)}, {random.uniform(0,1)}]"
        category = random.choice(categories)
        author = random.choice(authors)
        
        cursor.execute(
            "INSERT INTO documents (title, author, category, embedding) VALUES (%s, %s, %s, %s)",
            (f"Paper {i}", author, category, vec)
        )

    print("Database setup complete! Analyzing table to update planner statistics...")
    cursor.execute("ANALYZE documents;")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup_database()