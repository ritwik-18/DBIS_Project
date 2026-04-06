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
        
        # The ML Routing Logic
        if strictness >= 0.80:
            path = "PATH_B (Filter First - Exact Math)"
        else:
            path = "PATH_A (Vector First - HNSW)"
            
        print(f"[ROUTER] Filter: '{filter_query}' | Strictness: {strictness:.2f} | Action: {path}")

if __name__ == "__main__":
    proxy = HybridSearchProxy()
    dummy_vector = [0.5, 0.5, 0.5]

    print("\n--- TEST 1: Broad Filter (Computer Science) ---")
    proxy.search(dummy_vector, "category = %s", ('Computer Science',))

    print("\n--- TEST 2: Strict Filter (Rare Author) ---")
    proxy.search(dummy_vector, "author = %s", ('Srinivasa Ramanujan',))
    print("\n")