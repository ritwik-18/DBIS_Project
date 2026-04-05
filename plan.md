
# 🧠 Self-Optimizing Adaptive Hybrid Query Engine

## 🏗️ System Architecture

```
User Query
   ↓
Pre-Optimization Layer
   ↓
PostgreSQL + pgvector
   ↓
Post-Execution Advisor
```

---

## 🚀 Pre-Optimization Layer (Core System)

The **Pre-Optimization Layer** is the central intelligence of the system.
It intercepts queries before execution and dynamically determines the best execution strategy.

---

### 🔹 Level 1: ML Execution Router

**Goal:**
Decide whether a query should run:

* **Vector Search First (Post-filtering)**
* **Metadata Filter First (Pre-filtering)**

**Steps:**

* Intercept incoming SQL query
* Fetch statistics from `pg_stats` (estimate filter selectivity)
* Use ML model or rule-based logic to estimate execution cost
* Choose optimal execution strategy:

  * **Vector-first** for broad filters
  * **Filter-first** for highly selective filters
* Rewrite SQL query accordingly

**Output:**

* Optimized query plan before execution

---

### 🔹 Level 2: Adaptive ACORN Simulation

**Goal:**
Handle **moderate selectivity cases** where both strategies are inefficient.

**Steps:**

* Detect "moderate" selectivity range
* Apply adaptive ANN tuning:

  * Increase search depth:

    ```sql
    SET LOCAL hnsw.ef_search = 150–200;
    ```
  * Increase candidate pool size (k)
* Retry query if filtered results are insufficient

**Output:**

* Improved recall and performance without modifying database internals

---

### 🔹 Level 3: Autonomous DBA (Trigger Logic)

**Goal:**
Automatically optimize recurring query patterns.

**Steps:**

* Log query patterns and execution decisions
* Detect frequently occurring strict filters
* Mark them as optimization candidates
* Trigger index creation for repeated bottlenecks

Example:

```sql
CREATE INDEX idx_physics_docs
ON docs USING hnsw (embedding)
WHERE category = 'Physics';
```

**Output:**

* Self-improving database with workload-aware indexing

---

### 🔹 Level 4: Native Engine (Future Work)

**Goal:**
Move optimization logic inside the database engine.

**Planned Enhancements:**

* Replace external proxy with internal DB integration
* Use PostgreSQL `planner_hook`
* Inject execution strategies during query planning
* Implement ACORN-style traversal natively
* (Optional) Build custom embedding pipeline

**Note:**
This level is **not part of current implementation**.

---

### 📤 Pre-Optimization Output

* Optimized SQL query
* Tuned ANN parameters (e.g., `ef_search`, candidate size)

---

## ⚙️ Execution Layer

### PostgreSQL + pgvector

Responsible for executing the optimized query:

* Performs vector search using HNSW
* OR executes filter-first exact computation
* Applies runtime tuning (`ef_search`, etc.)

---

## 📊 Post-Execution Advisor (Learning Layer)

**Goal:**
Continuously improve system performance using feedback.

**Steps:**

* Capture `EXPLAIN ANALYZE`
* Record:

  * execution time
  * query plan
* Detect inefficiencies
* Identify repeated patterns
* Trigger structural optimizations:

  * Partial index creation

Example:

```sql
CREATE INDEX idx_filtered
ON docs USING hnsw (embedding)
WHERE <frequent_filter>;
```

**Output:**

* Continuous system improvement via feedback loop

---

## 🔁 Feedback Loop

The system forms a **closed optimization loop**:

1. Predict → Optimize → Execute
2. Observe → Learn → Improve

---

## 🧠 Summary

This architecture enables:

* **Dynamic query execution strategy selection**
* **Adaptive ANN search tuning**
* **Workload-aware indexing**
* **Continuous performance improvement**

---
