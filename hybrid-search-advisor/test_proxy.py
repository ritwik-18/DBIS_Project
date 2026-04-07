import psycopg2

class HybridSearchProxy:
    def __init__(self):
        # Connect using the new 5454 port
        self.conn = psycopg2.connect(
            host="localhost", database="hybrid_search", user="admin", password="password", port="5454"
        )

    def calculate_strictness(self, filter_query, params):
        with self.conn.cursor() as cur:
            cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = 'documents';")
            total_rows = cur.fetchone()[0]

            # Ask Postgres planner how many rows this filter will return
            cur.execute(f"EXPLAIN (FORMAT JSON) SELECT id FROM documents WHERE {filter_query}", params)
            plan = cur.fetchone()[0][0]['Plan']
            filtered_rows = plan['Plan Rows']

            return 1.0 - (filtered_rows / total_rows)

    def search(self, query_vector, filter_query, params):
        strictness = self.calculate_strictness(filter_query, params)
        
        with self.conn.cursor() as cur:
            if strictness >= 0.80:
                # 1. THE STRICT FILTER (>80% dropped)
                path = "PATH_B (Filter First - Exact Math)"
                # Actual execution would use the CTE query here
                
            elif strictness >= 0.30:
                # 2. THE DANGER ZONE (30% - 80% dropped)
                path = "PATH_A_TUNED (Danger Zone! Injecting ef_search=200)"
                
                # Dynamically altering PostgreSQL parameters for just this one transaction
                cur.execute("BEGIN;")
                cur.execute("SET LOCAL hnsw.ef_search = 200;")
                # Actual execution would run PATH_A_QUERY here
                cur.execute("COMMIT;")
                
            else:
                # 3. THE BROAD FILTER (<30% dropped)
                path = "PATH_A (Vector First - Standard HNSW)"
                # Actual execution would run normal PATH_A_QUERY here

        print(f"[ROUTER] Filter: {str(filter_query):<15} | Strictness: {strictness:.2f} | Action: {path}")

if __name__ == "__main__":
    proxy = HybridSearchProxy()
    dummy_vector = [0.5, 0.5, 0.5]

    print("\n--- TEST 1: Broad Filter (<30% Dropped) ---")
    proxy.search(dummy_vector, "category = %s", ('Computer Science',))

    print("\n--- TEST 2: THE DANGER ZONE (~50% Dropped) ---")
    # Using 'id > 500' literally slices the database in half
    proxy.search(dummy_vector, "id > %s", (500,))

    print("\n--- TEST 3: Strict Filter (>80% Dropped) ---")
    proxy.search(dummy_vector, "author = %s", ('Srinivasa Ramanujan',))
    print("\n")