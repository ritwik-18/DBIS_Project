"""
Bottleneck Detector

Analyzes PostgreSQL EXPLAIN ANALYZE outputs to identify slow
scans, expensive sorts, or poor query plans in hybrid searches.
"""

def analyze_execution_plan(explain_json: dict) -> list[str]:
    """
    Parses an execution plan to find performance bottlenecks.
    
    Returns:
        A list of identified issues (e.g., ["Sequential Scan on large table detected"]).
    """
    issues = []
    # TODO: Traverse the EXPLAIN JSON tree.
    # Check for nodes like 'Seq Scan' or high 'Total Cost'
    return issues