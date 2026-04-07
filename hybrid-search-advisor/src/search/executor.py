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
            # If the key already has an operator like > or <, don't add an equals sign
            if any(op in key for op in ['=', '<', '>']):
                clauses.append(f"{key} %s")
            else:
                clauses.append(f"{key} = %s")
            values.append(value)
            
        return " AND ".join(clauses), values

    def execute_hybrid_search(self, query_vector: list[float], filters: dict, top_k: int = 10):
        """
        The main proxy interceptor.
        """
        filter_clause, filter_values = self._build_filter_clause(filters)
        
        # 1. Evaluate strictness and get optimal path
        strictness = self.router.calculate_filter_strictness(filter_clause, tuple(filter_values))
        chosen_path = self.router.predict_optimal_path(strictness)
        
        print(f"[PROXY ALERT] Filter Strictness: {strictness:.2f} | Routing to: {chosen_path}")

        # 2. Format SQL and correctly order the parameters!
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        
        if chosen_path == "PATH_B":
            # PATH B order: Filter, Vector, Vector, Limit
            query_params = tuple(filter_values) + (vector_str, vector_str, top_k)
            sql = PATH_B_QUERY.format(filter_clause=filter_clause)
        else:
            # PATH A order: Vector, Filter, Vector, Limit
            query_params = (vector_str,) + tuple(filter_values) + (vector_str, top_k)
            sql = PATH_A_QUERY.format(filter_clause=filter_clause)

        # 3. Execute via the optimal path
        with self.conn.cursor() as cur:
            if chosen_path == "PATH_A_TUNED":
                # Dynamic ACORN Simulation: Inject higher search depth
                cur.execute("BEGIN;")
                cur.execute("SET LOCAL hnsw.ef_search = 200;")
                cur.execute(sql, query_params)
            else:
                cur.execute(sql, query_params)

            # 4. Fetch the results IMMEDIATELY
            results = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            formatted_results = [dict(zip(colnames, row)) for row in results]

            # 5. Clean up the transaction if we opened one for the Danger Zone
            if chosen_path == "PATH_A_TUNED":
                cur.execute("COMMIT;")

            return formatted_results