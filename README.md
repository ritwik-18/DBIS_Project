# Self-Optimizing Hybrid Document Search System

**Team:** Ritwik, Yasasvi, MVV Harsha, Ajay

A PostgreSQL-based hybrid document search system that supports semantic and metadata search. This project includes an outer-layer advisor that analyzes slow hybrid queries and recommends evidence-based optimizations like index creation or query restructuring.

## Project Structure
- `/db`: PostgreSQL and pgvector setup.
- `/src/ingestion`: Document parsing and vector embedding generation.
- `/src/search`: Semantic, metadata, and hybrid SQL execution.
- `/src/advisor`: Query analysis and optimization recommendations.