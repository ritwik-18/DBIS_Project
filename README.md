# ACORN-1 in pgvector

An implementation of **ACORN-1** (ANN Constraint-Optimized Retrieval Network) on top of pgvector, bringing predicate-aware hybrid search to PostgreSQL. Based on the paper *"ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data"* (Patel et al., 2024).

## What this is

Standard HNSW vector search treats the index as predicate-blind: it finds the nearest vectors, then filters by metadata afterward (post-filtering). For low-selectivity predicates or workloads with negative query correlation, this wastes most of its work on non-matching candidates and often fails to reach high recall.

ACORN-1 changes the search algorithm itself. At each visited node during graph traversal, instead of looking only at direct neighbors, it expands to two-hop neighbors and filters by predicate. Non-matching nodes act as "bridges" that connect predicate-matching regions of the graph that would otherwise be unreachable.

This repository ports the ACORN-1 search algorithm into pgvector's HNSW implementation, allowing PostgreSQL to perform predicate-pushdown ANN search via a single GUC variable.

## How it works

Construction is unchanged. ACORN-1 builds a standard HNSW index with `INCLUDE (label_column)` so the predicate column is stored in the index leaf tuples. At search time, setting `hnsw.acorn_filter_label` activates a modified greedy search that:

1. Reads element labels directly from index pages (no heap fetch needed)
2. Walks two-hop neighborhoods at each visited node
3. Filters candidates by the GUC label value during traversal
4. Truncates the candidate pool to size M before greedy expansion

The standard `HnswSearchLayer` is left untouched. ACORN-1 lives in a parallel function `HnswSearchLayerAcorn` that's only invoked when the GUC is set.

## Usage

```sql
-- Build the index with the predicate column included
CREATE INDEX idx_items_acorn ON items
USING hnsw (embedding vector_l2_ops)
INCLUDE (tag_id)
WITH (m = 32, ef_construction = 64);

-- Standard HNSW search (no predicate pushdown)
SELECT id FROM items
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;

-- ACORN-1 predicate-pushdown search
SET hnsw.acorn_filter_label = 5;
SELECT id FROM items
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
-- No WHERE clause — filtering happens inside the index traversal

-- Disable ACORN
RESET hnsw.acorn_filter_label;
```

## What's been changed in pgvector

### Storage layer
- `HnswElementTupleData` extended with a `labels[HNSW_HEAPTIDS]` array, parallel to the existing `heaptids[]` array
- Bumped `HNSW_VERSION` to 2 to mark the new tuple format
- Labels are read from the `INCLUDE` column at insert/build time and persisted in the index

### Search layer
- New GUC `hnsw.acorn_filter_label` (default -1, meaning ACORN inactive)
- New function `HnswSearchLayerAcorn` performing two-hop expansion with predicate filtering
- New helper `HnswLoadUnvisitedAcorn` for the two-hop neighbor expansion
- New helper `AcornElementMatches` for O(1) label checks against an element's heaptids
- Bottom-level scan in `GetScanItems` routes to the ACORN variant when the GUC is set; upper layers use the standard search

### Build & insert paths
- `InsertTuple` (build) and `HnswInsertTuple` (online) read the label from `values[1]` when an `INCLUDE` column is present
- `HnswAddHeapTid` extended to accept a label parameter, kept in lockstep with the heap TID
- `HnswSetElementTuple` and `HnswLoadElementFromTuple` propagate labels between memory and disk representations

## Benchmarking

The `Alpha.sql` script reproduces the SIFT1M benchmark from Section 7 of the paper:

- **Phase 0:** assigns random labels (1–12) to base and query vectors, ~8.3% selectivity
- **Phase 1:** generates filtered ground truth via brute-force scan (no indices)
- **Phase 2:** ACORN-1 with `hnsw.acorn_filter_label` set; no `WHERE` clause
- **Phase 3:** HNSW post-filter with executor-driven `WHERE tag_id = X`
- **Phase 4:** brute-force pre-filter (validates ground truth, perfect recall)
- **Phase 5:** index size and build time comparison

Each phase drops all indices and rebuilds only what it needs, so methods don't contaminate each other.

## Results on SIFT1M

Recall@10 vs QPS at varying `ef_search`:

| ef_search | ACORN-1 Recall | ACORN-1 QPS | Post-filter Recall | Post-filter QPS |
|-----------|---------------|-------------|---------------------|-----------------|
| 10 | 0.82 | ~80 | 0.09 | ~810 |
| 50 | 0.99 | ~21 | 0.42 | ~280 |
| 100 | 1.00 | ~11 | 0.78 | ~170 |
| 200 | 1.00 | ~4 | 0.99 | ~98 |
| 800 | 1.00 | ~1 | 1.00 | ~35 |

ACORN-1 reaches high recall (>0.99) at much lower `ef_search` than post-filter, confirming the algorithmic claim from the paper. At equivalent recall, post-filter has higher QPS in our PostgreSQL implementation — see "Why the QPS gap" below.

## Why the QPS gap vs the paper

The paper reports ACORN-1 achieving 2-10× higher QPS than post-filter at fixed recall. We see the inverse: post-filter is faster than ACORN-1 in absolute QPS at equivalent recall. This is **structural to PostgreSQL**, not a bug:

- The paper's reference implementation is in C++ inside FAISS, with vectors in flat in-memory arrays. Accessing a vector is one pointer dereference (~1 ns)
- Our implementation goes through PostgreSQL's buffer pool: every vector access requires `ReadBuffer` + `LockBuffer` + tuple extraction + `UnlockReleaseBuffer` (~1-5 µs even when the page is hot in the buffer pool)
- ACORN-1's two-hop expansion does roughly 10× more buffer accesses per visited node than standard HNSW. In FAISS this is invisible. In PostgreSQL it dominates everything else.

Distance computation, which the paper assumes is the bottleneck, is a tiny fraction of total cost in our setting. The algorithmic property (fewer distance computations to reach a given recall) still holds — it just doesn't translate to higher QPS because distance computation isn't the limiting factor.

## Optimizations applied

Three I/O redundancies were identified during development:

**Double-fetch (fixed):** the original implementation read each candidate's element page twice — once to check the label, once to compute the distance. Fixed by computing distance during the same `ReadBuffer` that reads the label, and passing the loaded element + distance through parallel arrays alongside the existing `unvisited[]` array. The `HnswUnvisited` union was left untouched.

**Hash thrashing (intentionally not fixed):** the local dedup hash is created and destroyed on every `HnswLoadUnvisitedAcorn` call. Could be amortized by passing a reusable hash from the caller. Left as future work because the fix has correctness implications around hash lifetime and `tmpCtx` interaction.

**Bridge re-reads across expansions (intentionally not fixed):** when two visited nodes share a non-matching bridge, the bridge gets re-read once per expansion. Cannot be safely fixed by globally inserting bridges into `v->tids` — that would make them unreachable as candidates from a different central node, breaking ACORN's correctness.

## Constraints

- The `INCLUDE` column must be a single `int4` column (the label)
- ACORN filtering uses exact equality only (matches the SIFT1M / Paper LCPS benchmark setup from the paper)
- Selectivity estimation is the user's responsibility — there's no automatic fallback to pre-filtering for very low selectivity (the paper recommends `s_min = 1/γ = 1/12` for SIFT1M)
- The implementation targets ACORN-1 only. ACORN-γ requires neighbor-list expansion during construction and is not implemented here

## Files modified

- `src/hnsw.h` — `HnswQuery` struct extended with `filter_label`/`has_filter`; `HnswElementData` and `HnswElementTupleData` carry parallel `labels[]` arrays; ACORN function declarations
- `src/hnsw.c` — `hnsw_acorn_filter_label` GUC registration
- `src/hnswutils.c` — `HnswSearchLayerAcorn`, `HnswLoadUnvisitedAcorn`, `AcornElementMatches`; updated `HnswSetElementTuple`, `HnswLoadElementFromTuple`, `HnswAddHeapTid`, `HnswInitElement`
- `src/hnswbuild.c` — label extraction in `InsertTuple` and `AddDuplicateInMemory`
- `src/hnswinsert.c` — label extraction in `HnswInsertTuple` and label preservation in `AddDuplicateOnDisk`
- `src/hnswscan.c` — `GetScanItems` and `ResumeScanItems` route bottom-level scan to `HnswSearchLayerAcorn` when ACORN is active; `hnswgettuple` filters non-matching heap TIDs from each returned element

## Building

Standard pgvector build:

```bash
cd pgvector-acorn
make
sudo make install
```

In PostgreSQL:

```sql
CREATE EXTENSION vector;
```

Indices built with this fork have `version = 2` in their metapage and are NOT compatible with stock pgvector. Indices built with stock pgvector are NOT readable by this fork (the tuple layout differs).

## Reference

Patel, L., Kraft, P., Guestrin, C., Zaharia, M. *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.* arXiv:2403.04871, 2024. https://arxiv.org/abs/2403.04871

## Status

Working implementation, validated against the paper's SIFT1M benchmark. Recall behavior matches the paper's description. QPS numbers are bounded by PostgreSQL's buffer pool architecture rather than the algorithm itself.
