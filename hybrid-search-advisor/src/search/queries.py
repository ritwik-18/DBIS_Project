"""
Execution Paths for Hybrid Search

Defines Path A (Vector First) and Path B (Filter First) SQL structures.
"""

# PATH A: Vector First (Post-Filtering)
# Relies on the HNSW index to quickly find semantic matches, then filters out bad IDs.
# Best when the filter keeps most of the data.
PATH_A_QUERY = """
    SELECT id, title, author, category, publish_date, 
           1 - (embedding <-> %s::vector) AS similarity
    FROM documents
    WHERE {filter_clause}
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
"""

# PATH B: Filter First (Pre-Filtering)
# Forces PostgreSQL to use standard B-Tree indexes to filter data first.
# Calculates exact mathematical distance (without HNSW) on the tiny remaining dataset.
# Best when the filter deletes > 80% of the data.
PATH_B_QUERY = """
    WITH filtered_docs AS (
        SELECT id, title, author, category, publish_date, embedding
        FROM documents
        WHERE {filter_clause}
    )
    SELECT id, title, author, category, publish_date, 
           1 - (embedding <-> %s::vector) AS similarity
    FROM filtered_docs
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
"""