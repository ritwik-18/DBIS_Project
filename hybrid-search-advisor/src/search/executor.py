"""
The Execution Proxy

Intercepts user queries, queries the ML model, and forces PostgreSQL down 
the statistically fastest execution path.
"""
from src.advisor.ml_router import MLExecutionRouter
from src.search.queries import PATH_A_QUERY, PATH_B_QUERY

class HybridSearchProxy:
    def __init__(self, connection):
        self.conn = connection
        self.router = MLExecutionRouter(connection)

    def _build_filter_clause(self, filters: dict) -> tuple[str, list]:
        """Converts a dictionary of filters into SQL WHERE clauses."""
        if not filters:
            return "1=1", []
            
        clauses = []
        values = []
        for key, value in filters.items():
            clauses.append(f"{key} = %s")
            values.append(value)
            
        return " AND ".join(clauses), values

    def execute_hybrid_search(self, query_vector: list[float], filters: dict, top_k: int = 10):
        """
        The main proxy interceptor.
        """
        filter_clause, filter_values = self._build_filter_clause(filters)
        
        # 1. Ask the ML Router to evaluate filter strictness
        strictness = self.router.calculate_filter_strictness(filter_clause, tuple(filter_values))
        
        # 2. ML Router predicts the optimal path
        chosen_path = self.router.predict_optimal_path(strictness)
        
        print(f"[PROXY ALERT] Filter Strictness: {strictness:.2f} | Routing to: {chosen_path}")

        # 3. Format the SQL based on the routing decision
        if chosen_path == "PATH_A":
            sql = PATH_A_QUERY.format(filter_clause=filter_clause)
        else:
            sql = PATH_B_QUERY.format(filter_clause=filter_clause)

        # 4. Prepare parameters (vector, vector again for ordering, limit)
        # Note: pgvector requires string representation of arrays
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        query_params = tuple(filter_values) + (vector_str, vector_str, top_k)

        # 5. Execute via the optimal path
        with self.conn.cursor() as cur:
            cur.execute(sql, query_params)
            results = cur.fetchall()
            
            # Fetch column names
            colnames = [desc[0] for desc in cur.description]
            
            # Return as list of dictionaries for clean API output
            return [dict(zip(colnames, row)) for row in results]

# Example usage (assuming DB connection is established):
# proxy = HybridSearchProxy(db_connection)
# results = proxy.execute_hybrid_search(query_vector=[0.1, 0.5, ...], filters={'category': 'Physics'})