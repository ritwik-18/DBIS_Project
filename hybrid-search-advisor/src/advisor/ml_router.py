"""
ML Execution Router

Team: Ritwik, Yasasvi, MVV
Calculates filter strictness and routes queries to the optimal execution path.
"""
import psycopg2

class MLExecutionRouter:
    def __init__(self, connection):
        self.conn = connection

    def calculate_filter_strictness(self, filter_query: str, params: tuple) -> float:
        """
        Calculates how much data the metadata filter will drop.
        Uses PostgreSQL's EXPLAIN planner to estimate row counts without running the query.
        """
        # Get total rows in the table
        with self.conn.cursor() as cur:
            cur.execute("SELECT reltuples::bigint AS estimate FROM pg_class WHERE relname = 'documents';")
            total_rows = cur.fetchone()[0]
            
            if total_rows == 0:
                return 0.0

            # Ask Postgres planner how many rows the filter will leave
            explain_sql = f"EXPLAIN (FORMAT JSON) SELECT id FROM documents WHERE {filter_query}"
            cur.execute(explain_sql, params)
            plan = cur.fetchone()[0][0]['Plan']
            filtered_rows = plan['Plan Rows']

            # Calculate strictness (0.0 = keeps everything, 1.0 = drops everything)
            strictness = 1.0 - (filtered_rows / total_rows)
            return strictness

    def predict_optimal_path(self, strictness: float) -> str:
        """
        The ML / Heuristic decision engine.
        - strictness < 0.8 (broad filter): Path A (Vector First / HNSW)
        - strictness >= 0.8 (strict filter): Path B (Filter First / Exact Math)
        
        Note: Future ML model will replace this threshold with a trained prediction 
        based on pg_stats and vector distribution.
        """
        THRESHOLD = 0.80 
        
        if strictness >= THRESHOLD:
            return "PATH_B"
        else:
            return "PATH_A"