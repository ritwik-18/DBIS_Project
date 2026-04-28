# ACORN-1 Filtered ANN Search — Implementation Notes

This document describes every source change made to add **ACORN-1 predicate-filtered approximate nearest-neighbour search** to the pgvector HNSW extension.  The implementation follows the algorithm described in:

> *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data* (Patel et al., 2024).

No existing function was modified in a breaking way. All new code is additive. The unfiltered code path is unchanged.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Model Changes](#data-model-changes)
3. [File-by-File Changes](#file-by-file-changes)
   - [hnsw.h](#hnswh)
   - [hnsw.c](#hnswc)
   - [hnswutils.c](#hnswutilsc)
   - [hnswbuild.c](#hnswbuildc)
   - [hnswinsert.c](#hnswinsertc)
   - [hnswscan.c](#hnswscanc)
4. [How the Filter Is Activated at Query Time](#how-the-filter-is-activated-at-query-time)
5. [Index DDL](#index-ddl)
6. [End-to-End Data Flow](#end-to-end-data-flow)
7. [Design Decisions & Invariants](#design-decisions--invariants)

---

## Overview

ACORN-1 performs filtered ANN by expanding **two hops** (the direct neighbours of a candidate node plus the neighbours of those neighbours) at graph layer 0, admitting only candidates whose stored label matches the query predicate.  Non-matching hop-1 nodes are still traversed as *bridges* — that is the core insight of the paper.

The implementation requires three things:

1. **Storage** — each heap TID stored in an element must carry its integer label.
2. **Build / insert** — the label must be read from the index `INCLUDE` column and written into the element tuple.
3. **Search** — a new `HnswSearchLayerAcorn` function replaces `HnswSearchLayer` at layer 0 when a filter is active.

---

## Data Model Changes

### Index version bump

```c
// hnsw.h
#define HNSW_VERSION  2   /* was 1 — ACORN: per-heaptid labels added */
```

The on-disk format of `HnswElementTupleData` changes, so a new version number prevents an older binary from reading a label-aware index.

### Per-TID label array

Every element can hold up to `HNSW_HEAPTIDS` (= 10) heap TIDs for non-HOT update deduplication.  A **parallel `int32 labels[HNSW_HEAPTIDS]`** array is added alongside `heaptids` in both the in-memory struct and the on-disk tuple:

```c
// hnsw.h — HnswElementData (in-memory)
ItemPointerData heaptids[HNSW_HEAPTIDS];
int32           labels[HNSW_HEAPTIDS];   /* ACORN: parallel to heaptids */

// hnsw.h — HnswElementTupleData (on-disk)
ItemPointerData heaptids[HNSW_HEAPTIDS];
int32           labels[HNSW_HEAPTIDS];   /* ACORN: parallel to heaptids */
```

A label of `HNSW_LABEL_NONE` (`INT32_MIN`) means "no label assigned" and never matches any real filter.

### HnswQuery extended

```c
// hnsw.h — HnswQuery
typedef struct HnswQuery {
    Datum   value;
    int32   filter_label;   /* ACORN: label to match */
    bool    has_filter;     /* ACORN: true when filtering active */
} HnswQuery;
```

`has_filter` is the zero-cost gate: when `false`, every code path that inspects `filter_label` is skipped entirely.

---

## File-by-File Changes

### `hnsw.h`

| What changed | Why |
|---|---|
| `HNSW_VERSION` bumped to `2` | On-disk format change (labels array added to element tuple) |
| `HNSW_LABEL_NONE` sentinel (`INT32_MIN`) defined | Safe default that never matches a real filter |
| `int32 labels[HNSW_HEAPTIDS]` added to `HnswElementData` | In-memory element carries per-TID labels |
| `int32 labels[HNSW_HEAPTIDS]` added to `HnswElementTupleData` | On-disk element tuple carries per-TID labels |
| `HnswQuery` extended with `filter_label` + `has_filter` | Query carries its predicate through the search stack |
| `HnswAddHeapTid` signature changed: `(element, heaptid, int32 label)` | Label must travel with the TID at every call site |
| `HnswInsertTupleOnDisk` signature changed: `(..., int32 label)` | Label passed through from insert / build |
| `AcornElementMatches` declared | O(1) label check exported for use in `hnswscan.c` |
| `HnswSearchLayerAcorn` declared | New filtered search function |
| `extern int hnsw_acorn_filter_label` declared | GUC variable shared between `hnsw.c` and `hnswscan.c` |

---

### `hnsw.c`

**Location:** `HnswInit()`

```c
int  hnsw_acorn_filter_label = -1;   /* global, default = disabled */

DefineCustomIntVariable(
    "hnsw.acorn_filter_label",
    "Integer label for ACORN-1 filtered ANN search. -1 disables filtering.",
    NULL,
    &hnsw_acorn_filter_label,
    -1, -1, INT_MAX,
    PGC_USERSET, 0, NULL, NULL, NULL);
```

A single GUC (`hnsw.acorn_filter_label`) controls filtering at the session level:

```sql
SET hnsw.acorn_filter_label = 42;   -- enable filter for label 42
SET hnsw.acorn_filter_label = -1;   -- disable (default)
```

No other code in `hnsw.c` is changed.

---

### `hnswutils.c`

This file contains the three new/modified functions that are the core of the implementation.

#### 1. `HnswAddHeapTid` — signature extended

```c
// Before
void HnswAddHeapTid(HnswElement element, ItemPointer heaptid)

// After
void HnswAddHeapTid(HnswElement element, ItemPointer heaptid, int32 label)
{
    element->labels[element->heaptidsLength] = label;  // store label in lockstep
    element->heaptids[element->heaptidsLength++] = *heaptid;
}
```

`HnswInitElement` seeds the first slot with `HNSW_LABEL_NONE`; callers that know the label overwrite it immediately after.

#### 2. `HnswLoadElementFromTuple` — labels round-tripped through disk tuple

```c
// Serialise (HnswSetElementTuple path)
for (int i = 0; i < HNSW_HEAPTIDS; i++)
    etup->labels[i] = (i < element->heaptidsLength)
                      ? element->labels[i]
                      : HNSW_LABEL_NONE;

// Deserialise (HnswLoadElementFromTuple)
HnswAddHeapTid(element, &etup->heaptids[i], etup->labels[i]);
// ↑ was: HnswAddHeapTid(element, &etup->heaptids[i])
```

Labels survive a page flush / reload cycle transparently.

#### 3. `AcornElementMatches` — new function

```c
bool
AcornElementMatches(HnswElement element, int32 filter_label)
{
    for (int i = 0; i < element->heaptidsLength; i++)
        if (element->labels[i] == filter_label)
            return true;
    return false;
}
```

An element matches if **any** of its stored TIDs carries the requested label. This correctly handles the non-HOT-update dedup case where one graph node covers multiple physical heap rows.

#### 4. `HnswLoadUnvisitedAcorn` — new static function (two-hop expansion)

This replaces `HnswLoadUnvisitedFromDisk` in the ACORN search path.

**Algorithm (paper §5.3):**
1. Load the direct neighbours of the current candidate (`hop1Tids`).
2. For each hop-1 node:
   - If it already appears in the global visited hash → skip.
   - Load its element tuple to read its label.
   - If it matches the filter → add to the output pool.
   - **Regardless of match**, walk its neighbours (`hop2Tids`) and add any matching ones to the pool. Non-matching hop-1 nodes act as bridges.
3. Output is capped at `lm` to prevent buffer overflow.
4. A per-call **local dedup hash** prevents a node being returned twice when it is reachable as both a hop-1 and a hop-2 neighbour.
5. Global visited marking happens **after** this function returns (delayed commit in `HnswSearchLayerAcorn`) so duplicate detection across loop iterations still works.

```c
static void
HnswLoadUnvisitedAcorn(HnswElement element, HnswUnvisited *unvisited,
                       int *unvisitedLength, visited_hash *v,
                       Relation index, int m, int lm, int lc,
                       int32 filter_label)
```

#### 5. `HnswSearchLayerAcorn` — new exported function

A near-identical copy of `HnswSearchLayer`. **The only controlled difference:** in the disk path it calls `HnswLoadUnvisitedAcorn` instead of `HnswLoadUnvisitedFromDisk`.

```c
// Disk path — inside the main search loop
HnswLoadUnvisitedAcorn(cElement, unvisited, &unvisitedLength,
                       v, index, m, lm, lc,
                       q->filter_label);   // ← only new argument
```

The in-memory path (used during index builds) falls through to the standard unfiltered expansion — ACORN filtering is query-time only.

**Delayed commit:** Global visited marking is done right before evaluating each candidate (not inside `HnswLoadUnvisitedAcorn`). This matches the invariant in `HnswSearchLayer` and correctly deduplicates candidates that appear in multiple two-hop expansions.

`HnswSearchLayer` itself is **never modified**.

#### Insert path initialisation

```c
// In HnswFindElementNeighbors (called during insert + build)
q.has_filter   = false;    // insert path never filters
q.filter_label = -1;
```

---

### `hnswbuild.c`

#### `InsertTuple` — reads label from INCLUDE column

```c
int32 label;

if (buildstate->indexInfo->ii_NumIndexAttrs >= 2 && !isnull[1])
    label = DatumGetInt32(values[1]);
else
    label = HNSW_LABEL_NONE;

// ... then passed to HnswInsertTupleOnDisk or stored on element:
element->labels[0] = label;
```

The label is sourced from the first `INCLUDE` column of the index — no heap re-fetch is needed because the index AM callback already receives all indexed and included column values.

#### `AddDuplicateInMemory` — forwards label to `HnswAddHeapTid`

```c
// Before
HnswAddHeapTid(dup, &element->heaptids[0]);

// After
HnswAddHeapTid(dup, &element->heaptids[0], element->labels[0]);
```

When a duplicate vector is found during the in-memory build phase, the new row's label travels with its TID into the existing element.

---

### `hnswinsert.c`

#### `HnswInsertTuple` — reads label from INCLUDE column

Same pattern as `hnswbuild.c`:

```c
int32 label;
if (RelationGetNumberOfAttributes(index) >= 2 && !isnull[1])
    label = DatumGetInt32(values[1]);
else
    label = HNSW_LABEL_NONE;

HnswInsertTupleOnDisk(index, &support, value, heaptid, false, label);
```

#### `HnswInsertTupleOnDisk` — stores label on element

```c
element = HnswInitElement(...);
element->labels[0] = label;   // ACORN: set label on element
```

---

### `hnswscan.c`

#### `GetScanItems` — activates filter from GUC

```c
q->has_filter    = false;
q->filter_label  = HNSW_LABEL_NONE;

if (hnsw_acorn_filter_label >= 0)
{
    q->has_filter   = true;
    q->filter_label = hnsw_acorn_filter_label;
}

// ... upper layer traversal unchanged (HnswSearchLayer, ef=1) ...

// Bottom layer (L = 0)
if (q->has_filter)
    return HnswSearchLayerAcorn(...);

return HnswSearchLayer(...);   // unfiltered path unchanged
```

Upper layers always use the unfiltered `HnswSearchLayer` with `ef = 1` (Stage 1 of the paper: descend to a good starting point). Only layer 0 switches to the ACORN variant.

#### `ResumeScanItems` — mirrors `GetScanItems` for iterative scan

```c
if (so->q.has_filter)
    return HnswSearchLayerAcorn(..., initVisited=false, ...);

return HnswSearchLayer(..., initVisited=false, ...);
```

`so->q.has_filter` is set once in `GetScanItems` and remains valid for the entire scan lifetime, so the iterative-scan resume path automatically routes through the same variant.

#### `hnswgettuple` — per-TID label check when draining results

```c
while (element->heaptidsLength > 0)
{
    int idx = element->heaptidsLength - 1;

    if (!so->q.has_filter || element->labels[idx] == so->q.filter_label)
    {
        heaptid = &element->heaptids[idx];
        element->heaptidsLength--;
        break;
    }
    /* Discard non-matching TID */
    element->heaptidsLength--;
}
```

When an element carries multiple TIDs (non-HOT updates), only those whose label matches the filter are returned to the executor.  This is the final safety net that keeps the unfiltered graph nodes from leaking incorrect rows into the result set.

---

## How the Filter Is Activated at Query Time

```sql
-- Activate for the session (or transaction)
SET hnsw.acorn_filter_label = 42;

-- Normal KNN query — now returns only rows whose indexed label = 42
SELECT * FROM items ORDER BY embedding <-> '[1,2,3]' LIMIT 10;

-- Deactivate
SET hnsw.acorn_filter_label = -1;
```

The GUC is `PGC_USERSET`, so it can be set per-transaction and is safe for connection poolers that use `SET LOCAL`.

---

## Index DDL

The label must be stored in the index via a PostgreSQL `INCLUDE` column:

```sql
-- 'tag' is an integer column on the table
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) INCLUDE (tag);
```

The index AM callback receives `values[0]` = the vector and `values[1]` = the tag integer.  No heap re-fetch is needed.

For backwards compatibility, if no `INCLUDE` column is present (`ii_NumIndexAttrs < 2` or `isnull[1]`), the label is stored as `HNSW_LABEL_NONE` and the node never matches any filter.

---

## End-to-End Data Flow

```
INSERT / COPY
  └─ hnswinsert.c :: HnswInsertTuple
       reads values[1] → label
       └─ HnswInsertTupleOnDisk(..., label)
            element->labels[0] = label
            └─ UpdateGraphOnDisk → HnswSetElementTuple
                 etup->labels[i] = element->labels[i]   ← written to page

BUILD
  └─ hnswbuild.c :: InsertTuple
       reads values[1] → label
       element->labels[0] = label
       └─ (in-memory) element lives in graphCtx with label
          (on-disk)  HnswInsertTupleOnDisk(..., label)

SEARCH  (SET hnsw.acorn_filter_label = N)
  └─ hnswscan.c :: GetScanItems
       q.has_filter = true; q.filter_label = N
       upper layers: HnswSearchLayer(ef=1)       ← unchanged
       layer 0:      HnswSearchLayerAcorn(ef_search)
                       └─ HnswLoadUnvisitedAcorn
                            hop-1: load element, check AcornElementMatches
                            hop-2: traverse through non-matching bridges
                       returns candidate list
  └─ hnswscan.c :: hnswgettuple
       per-TID label check before returning to executor
```

---

## Design Decisions & Invariants

| Decision | Rationale |
|---|---|
| Labels stored in-element, not a side table | No join at query time; labels are co-located with the TIDs already loaded during graph traversal |
| `HNSW_LABEL_NONE = INT32_MIN` | Out-of-band value that cannot be produced by a real `integer` column; safe default for unlabelled rows |
| Upper layers use unfiltered search | Paper §6.3.2 Stage 1: descend quickly to a good starting region before applying the predicate subgraph walk at L0 |
| `HnswSearchLayer` is never modified | Zero risk of regression on the unfiltered path; the ACORN variant is an additive parallel implementation |
| Local dedup hash inside `HnswLoadUnvisitedAcorn` | Prevents the same TID being returned twice from a single two-hop expansion without polluting the global visited hash prematurely |
| Output pool capped at `lm` | Prevents stack/heap overflow when `m` is large and both hop depths are full |
| `has_filter` boolean gate | When `false`, all ACORN branches are skipped with no runtime cost on unfiltered queries |
