-- =============================================================================
-- ACORN-1 SIFT1M Benchmark — Clean Implementation
-- =============================================================================
--
-- Reproduces ACORN paper Section 7, Figure 7a, Table 3
--
-- Three methods tested, each in complete isolation:
--
--   METHOD 1: ACORN-1
--     Index:  HNSW with INCLUDE(tag_id) — stores tag in index leaf pages
--     Search: SET hnsw.acorn_filter_label = <tag>
--             SELECT ... ORDER BY embedding <-> query LIMIT 10
--             (NO WHERE clause — filtering happens inside index traversal)
--     What happens: During greedy HNSW traversal, the index skips nodes
--             that fail the predicate and expands to two-hop neighbors
--             to find predicate-passing paths. This is the core ACORN idea.
--
--   METHOD 2: HNSW Post-filter
--     Index:  Plain HNSW (no metadata awareness)
--     Search: SELECT ... WHERE tag_id = <tag>
--             ORDER BY embedding <-> query LIMIT 10
--     What happens: HNSW returns candidates in distance order.
--             The executor checks each row against WHERE tag_id = <tag>.
--             Non-matching rows are discarded. Executor keeps pulling
--             from the index until 10 matching rows accumulate.
--             At selectivity ~0.083, it must examine ~12x more candidates.
--
--   METHOD 3: Pre-filter (brute force)
--     Index:  NONE — all indices disabled
--     Search: SELECT ... WHERE tag_id = <tag>
--             ORDER BY embedding <-> query LIMIT 10
--     What happens: Sequential scan over all 1M rows. Only rows with
--             matching tag_id survive the filter. Then exact distance
--             sort over the ~83K surviving rows. Perfect recall, slow.
--
-- Paper setup (Section 7.1.1, 7.2):
--   - 1M base vectors, 128-dim SIFT descriptors
--   - Random label 1-12 per vector (selectivity ~0.083)
--   - M=32, ef_construction=40
--   - Search: vary ef_search from 10 to 800, measure Recall@10 vs QPS
--
-- Prerequisites:
--   sift_base  (id, embedding vector(128), tag_id int)
--   sift_query (id, embedding vector(128), tag_id int)
--   Custom pgvector fork with hnsw.acorn_filter_label GUC
-- =============================================================================

\timing on
SET client_min_messages = 'notice';


-- #############################################################################
-- PHASE 0: DATA PREPARATION
-- #############################################################################

\echo ''
\echo '================================================================'
\echo 'PHASE 0: Data Preparation'
\echo '================================================================'

-- 0a. Assign random labels 1-12 to any untagged rows
DO $$
DECLARE
    cnt bigint;
BEGIN
    SELECT COUNT(*) INTO cnt FROM sift_base WHERE tag_id IS NULL;
    IF cnt > 0 THEN
        RAISE NOTICE 'Assigning random labels to % sift_base rows...', cnt;
        UPDATE sift_base SET tag_id = floor(random() * 12 + 1)::int WHERE tag_id IS NULL;
    ELSE
        RAISE NOTICE 'sift_base: all rows already tagged';
    END IF;

    SELECT COUNT(*) INTO cnt FROM sift_query WHERE tag_id IS NULL;
    IF cnt > 0 THEN
        RAISE NOTICE 'Assigning random labels to % sift_query rows...', cnt;
        UPDATE sift_query SET tag_id = floor(random() * 12 + 1)::int WHERE tag_id IS NULL;
    ELSE
        RAISE NOTICE 'sift_query: all rows already tagged';
    END IF;
END $$;

VACUUM ANALYZE sift_base;
VACUUM ANALYZE sift_query;

-- 0b. Sample 200 query vectors for benchmark
DROP TABLE IF EXISTS sift_query_sample CASCADE;

CREATE TABLE sift_query_sample AS
SELECT * FROM sift_query ORDER BY random() LIMIT 200;

CREATE INDEX ON sift_query_sample (id);

ANALYZE sift_query_sample;

\echo 'Query sample created:'
SELECT COUNT(*) AS num_queries FROM sift_query_sample;


-- #############################################################################
-- PHASE 1: FILTERED GROUND TRUTH (brute-force exact answers)
-- #############################################################################
--
-- For each of the 200 query vectors, find the TRUE 10 nearest neighbors
-- among base vectors that share the same tag_id. This is done via
-- sequential scan with no index — guaranteeing exact results.
--
-- Expected output: 200 queries x 10 neighbors = 2000 rows

\echo ''
\echo '================================================================'
\echo 'PHASE 1: Generate Filtered Ground Truth (brute-force exact KNN)'
\echo '================================================================'

-- Drop ALL indices to guarantee brute-force scan
DROP INDEX IF EXISTS idx_sift_acorn;
DROP INDEX IF EXISTS idx_sift_hnsw_plain;

SET enable_indexscan  = off;
SET enable_bitmapscan = off;
SET enable_seqscan    = on;

DROP TABLE IF EXISTS sift_filtered_gt CASCADE;

CREATE TABLE sift_filtered_gt AS
SELECT
    q.id        AS query_id,
    q.tag_id    AS query_tag,
    gt.base_id,
    gt.distance,
    gt.rank
FROM sift_query_sample q,
LATERAL (
    SELECT
        b.id AS base_id,
        (q.embedding <-> b.embedding) AS distance,
        ROW_NUMBER() OVER (ORDER BY (q.embedding <-> b.embedding)) AS rank
    FROM sift_base b
    WHERE b.tag_id = q.tag_id          -- predicate filter
    ORDER BY (q.embedding <-> b.embedding)
    LIMIT 10
) gt;

CREATE INDEX ON sift_filtered_gt (query_id);
ANALYZE sift_filtered_gt;

-- Restore planner defaults
SET enable_indexscan  = on;
SET enable_bitmapscan = on;

\echo 'Ground truth rows (should be 2000):'
SELECT COUNT(*) AS gt_rows FROM sift_filtered_gt;

\echo 'Sanity check — ground truth per query (should all be 10):'
SELECT
    MIN(cnt) AS min_per_query,
    MAX(cnt) AS max_per_query,
    AVG(cnt)::numeric(4,1) AS avg_per_query
FROM (SELECT query_id, COUNT(*) AS cnt FROM sift_filtered_gt GROUP BY query_id) t;


-- #############################################################################
-- PHASE 2: METHOD 1 — ACORN-1
-- #############################################################################
--
-- HOW IT WORKS:
--   1. Build HNSW index with INCLUDE(tag_id)
--      → tag_id is stored in the index leaf tuples
--   2. At search time, SET hnsw.acorn_filter_label = <desired_tag>
--   3. Query: ORDER BY embedding <-> query_vec LIMIT 10
--      → NO WHERE clause — the index itself filters during traversal
--   4. Inside the index: at each node, the traversal checks tag_id
--      against hnsw.acorn_filter_label. Non-matching neighbors are
--      skipped, and two-hop neighbors are explored instead.
--      This is the "predicate subgraph traversal" from the paper.
--
-- WHY NO WHERE CLAUSE:
--   If we added WHERE tag_id = X, PostgreSQL's executor would
--   apply it AFTER the index returns rows — that's post-filtering.
--   ACORN-1's whole point is filtering INSIDE the index traversal.
--   The GUC pushes the predicate into the HNSW scan operator.

\echo ''
\echo '================================================================'
\echo 'PHASE 2: METHOD 1 — ACORN-1 (predicate subgraph traversal)'
\echo '================================================================'

-- Clean slate: drop any existing indices
DROP INDEX IF EXISTS idx_sift_acorn;
DROP INDEX IF EXISTS idx_sift_hnsw_plain;

-- Build ACORN-1 index
SET maintenance_work_mem = '4GB';

\echo 'Building ACORN-1 index (HNSW with INCLUDE tag_id)...'
\timing on

CREATE INDEX idx_sift_acorn
ON sift_base
USING hnsw (embedding vector_l2_ops)
INCLUDE (tag_id)
WITH (m = 32, ef_construction = 64);

\timing off

-- Force index usage
SET enable_seqscan    = off;
SET enable_bitmapscan = off;
SET enable_indexscan  = on;

DO $$
DECLARE
    ef_val       int;
    num_queries  int;
    total_recall numeric;
    avg_recall   numeric;
    start_ts     timestamptz;
    end_ts       timestamptz;
    elapsed_sec  numeric;
    qps          numeric;
    q            record;
    recall_q     int;
BEGIN
    SELECT COUNT(*) INTO num_queries FROM sift_query_sample;

    -----------------------------------------------------------------
    -- Warmup: prime buffer cache
    -----------------------------------------------------------------
    EXECUTE 'SET hnsw.ef_search = 10';
    EXECUTE 'SET hnsw.acorn_filter_label = 1';
    FOR q IN SELECT id, tag_id, embedding FROM sift_query_sample LOOP
        PERFORM id FROM sift_base ORDER BY embedding <-> q.embedding LIMIT 1;
    END LOOP;

    RAISE NOTICE '';
    RAISE NOTICE '--- ACORN-1 Results ---';
    RAISE NOTICE 'ef_search | Recall@10 | QPS';
    RAISE NOTICE '----------|-----------|--------';

    -----------------------------------------------------------------
    -- Sweep ef_search values
    -----------------------------------------------------------------
    FOREACH ef_val IN ARRAY ARRAY[10, 50, 100, 200, 400, 600, 800]
    LOOP
        EXECUTE format('SET hnsw.ef_search = %s', ef_val);
        total_recall := 0;
        start_ts := clock_timestamp();

        FOR q IN SELECT id, tag_id, embedding FROM sift_query_sample
        LOOP
            -- Push predicate INTO the index traversal
            EXECUTE format('SET hnsw.acorn_filter_label = %s', q.tag_id);

            -- Query: NO WHERE clause
            -- The index does the filtering during greedy search
            SELECT COUNT(*) INTO recall_q
            FROM (
                SELECT id
                FROM sift_base
                ORDER BY embedding <-> q.embedding
                LIMIT 10
            ) acorn_result
            JOIN sift_filtered_gt gt
              ON gt.query_id = q.id
             AND gt.base_id  = acorn_result.id;

            total_recall := total_recall + recall_q;
        END LOOP;

        end_ts := clock_timestamp();
        elapsed_sec := EXTRACT(EPOCH FROM (end_ts - start_ts));
        avg_recall  := total_recall / (num_queries * 10.0);
        qps         := num_queries / elapsed_sec;

        RAISE NOTICE '    % | %  | %',
            lpad(ef_val::text, 5),
            to_char(avg_recall, '0.0000'),
            to_char(qps, '99999.0');
    END LOOP;

    -- Reset ACORN GUC
    EXECUTE 'RESET hnsw.acorn_filter_label';
    EXECUTE 'RESET hnsw.ef_search';
END $$;

-- Restore planner defaults before next phase
SET enable_seqscan    = on;
SET enable_bitmapscan = on;


-- #############################################################################
-- PHASE 3: METHOD 2 — HNSW POST-FILTER
-- #############################################################################
--
-- HOW IT WORKS:
--   1. Build a PLAIN HNSW index (no INCLUDE, no metadata awareness)
--   2. Query: WHERE tag_id = <tag> ORDER BY embedding <-> query LIMIT 10
--   3. The HNSW index returns candidates in distance order, one at a time.
--      The PostgreSQL executor evaluates WHERE tag_id = <tag> on each row.
--      Rows that fail the predicate are discarded.
--      The executor keeps pulling more rows from the index until
--      10 matching rows are collected (or the index is exhausted).
--   4. With selectivity ~1/12, roughly 12 candidates must be examined
--      per matching result. This wastes distance computations on
--      non-matching rows — the core inefficiency of post-filtering.
--
-- WHY WHERE CLAUSE IS NEEDED:
--   Unlike ACORN-1, the plain HNSW has no predicate awareness.
--   Filtering must happen OUTSIDE the index, in the executor.
--   The WHERE clause is the mechanism for this.
--
-- IMPORTANT: The ACORN index must be DROPPED first.
--   If both indices exist, the planner might pick the ACORN index
--   (which has INCLUDE tag_id and could evaluate WHERE cheaply),
--   giving ACORN results instead of true post-filter results.

\echo ''
\echo '================================================================'
\echo 'PHASE 3: METHOD 2 — HNSW Post-filter (executor-driven filtering)'
\echo '================================================================'

-- Drop ACORN index — only plain HNSW should exist
DROP INDEX IF EXISTS idx_sift_acorn;
DROP INDEX IF EXISTS idx_sift_hnsw_plain;

\echo 'Building plain HNSW index (no metadata)...'
\timing on

CREATE INDEX idx_sift_hnsw_plain
ON sift_base
USING hnsw (embedding vector_l2_ops)
WITH (m = 32, ef_construction = 64);

\timing off

-- Force index usage, disable ACORN
SET enable_seqscan    = off;
SET enable_bitmapscan = off;
SET enable_indexscan  = on;
SET hnsw.acorn_filter_label = -1;   -- Ensure ACORN logic is OFF

DO $$
DECLARE
    ef_val       int;
    num_queries  int;
    total_recall numeric;
    avg_recall   numeric;
    start_ts     timestamptz;
    end_ts       timestamptz;
    elapsed_sec  numeric;
    qps          numeric;
    q            record;
    recall_q     int;
BEGIN
    SELECT COUNT(*) INTO num_queries FROM sift_query_sample;

    -----------------------------------------------------------------
    -- Warmup
    -----------------------------------------------------------------
    EXECUTE 'SET hnsw.ef_search = 10';
    FOR q IN SELECT id, tag_id, embedding FROM sift_query_sample LOOP
        PERFORM id FROM sift_base
        WHERE tag_id = q.tag_id
        ORDER BY embedding <-> q.embedding
        LIMIT 1;
    END LOOP;

    RAISE NOTICE '';
    RAISE NOTICE '--- HNSW Post-filter Results ---';
    RAISE NOTICE 'ef_search | Recall@10 | QPS';
    RAISE NOTICE '----------|-----------|--------';

    -----------------------------------------------------------------
    -- Sweep ef_search values
    -----------------------------------------------------------------
    FOREACH ef_val IN ARRAY ARRAY[10, 50, 100, 200, 400, 600, 800]
    LOOP
        EXECUTE format('SET hnsw.ef_search = %s', ef_val);
        total_recall := 0;
        start_ts := clock_timestamp();

        FOR q IN SELECT id, tag_id, embedding FROM sift_query_sample
        LOOP
            -- Post-filter: WHERE clause is outside the index.
            -- HNSW returns nearest vectors regardless of tag.
            -- Executor discards non-matching rows, keeps pulling
            -- until 10 matches are found.
            SELECT COUNT(*) INTO recall_q
            FROM (
                SELECT id
                FROM sift_base
                WHERE tag_id = q.tag_id               -- executor-side filter
                ORDER BY embedding <-> q.embedding     -- HNSW distance order
                LIMIT 10
            ) postfilter_result
            JOIN sift_filtered_gt gt
              ON gt.query_id = q.id
             AND gt.base_id  = postfilter_result.id;

            total_recall := total_recall + recall_q;
        END LOOP;

        end_ts := clock_timestamp();
        elapsed_sec := EXTRACT(EPOCH FROM (end_ts - start_ts));
        avg_recall  := total_recall / (num_queries * 10.0);
        qps         := num_queries / elapsed_sec;

        RAISE NOTICE '    % | %  | %',
            lpad(ef_val::text, 5),
            to_char(avg_recall, '0.0000'),
            to_char(qps, '99999.0');
    END LOOP;

    EXECUTE 'RESET hnsw.ef_search';
    EXECUTE 'RESET hnsw.acorn_filter_label';
END $$;

SET enable_seqscan    = on;
SET enable_bitmapscan = on;


-- #############################################################################
-- PHASE 4: METHOD 3 — PRE-FILTER (Brute-Force Baseline)
-- #############################################################################
--
-- HOW IT WORKS:
--   1. NO index is used — all index scans are disabled
--   2. Query: WHERE tag_id = <tag> ORDER BY embedding <-> query LIMIT 10
--   3. PostgreSQL does a sequential scan over all 1M rows.
--      The WHERE clause filters down to ~83K matching rows (1/12).
--      Then it computes exact L2 distance for every matching row.
--      Then it sorts and returns the top 10.
--   4. This always achieves perfect recall (it's exact search).
--      The cost is proportional to the filtered set size: O(s * n).
--
-- WHY THIS IS SLOW:
--   For selectivity 0.083 on 1M vectors, it computes ~83,000
--   distance operations per query. Compare to ACORN-1 which needs
--   only ~1,000 (Table 3 in the paper).
--
-- THERE IS NO ef_search PARAMETER:
--   This is brute-force — no index, no tuning knob. Just one
--   measurement point: Recall = 1.0 at whatever QPS it achieves.

\echo ''
\echo '================================================================'
\echo 'PHASE 4: METHOD 3 — Pre-filter (brute-force exact search)'
\echo '================================================================'

-- Drop all vector indices — force sequential scan
DROP INDEX IF EXISTS idx_sift_acorn;
DROP INDEX IF EXISTS idx_sift_hnsw_plain;

SET enable_indexscan  = off;
SET enable_bitmapscan = off;
SET enable_seqscan    = on;

DO $$
DECLARE
    num_queries  int;
    total_recall numeric := 0;
    avg_recall   numeric;
    start_ts     timestamptz;
    end_ts       timestamptz;
    elapsed_sec  numeric;
    qps          numeric;
    q            record;
    recall_q     int;
BEGIN
    SELECT COUNT(*) INTO num_queries FROM sift_query_sample;

    RAISE NOTICE '';
    RAISE NOTICE '--- Pre-filter (Brute-Force) Results ---';

    start_ts := clock_timestamp();

    FOR q IN SELECT id, tag_id, embedding FROM sift_query_sample
    LOOP
        -- Pre-filter: sequential scan, filter by tag, exact distance sort
        -- No index involved — this is the brute-force baseline
        SELECT COUNT(*) INTO recall_q
        FROM (
            SELECT id
            FROM sift_base
            WHERE tag_id = q.tag_id               -- pre-filter: keep only matching tags
            ORDER BY embedding <-> q.embedding     -- brute-force distance over filtered set
            LIMIT 10
        ) prefilter_result
        JOIN sift_filtered_gt gt
          ON gt.query_id = q.id
         AND gt.base_id  = prefilter_result.id;

        total_recall := total_recall + recall_q;
    END LOOP;

    end_ts := clock_timestamp();
    elapsed_sec := EXTRACT(EPOCH FROM (end_ts - start_ts));
    avg_recall  := total_recall / (num_queries * 10.0);
    qps         := num_queries / elapsed_sec;

    RAISE NOTICE 'Recall@10 = %  (should be 1.0000 — this validates ground truth)',
        to_char(avg_recall, '0.0000');
    RAISE NOTICE 'QPS       = %', to_char(qps, '99999.0');
    RAISE NOTICE '';
    RAISE NOTICE 'If Recall is not 1.0, your ground truth has a bug.';
END $$;

SET enable_indexscan  = on;
SET enable_bitmapscan = on;


-- #############################################################################
-- PHASE 5: CONSTRUCTION METRICS
-- #############################################################################
--
-- Rebuild both indices to measure and compare their sizes.
-- Paper reference: Table 4 (TTI), Table 5 (Index Size)

\echo ''
\echo '================================================================'
\echo 'PHASE 5: Construction Metrics'
\echo '================================================================'
\echo ''
\echo 'Paper reference values for SIFT1M (96 vCPU AWS m5d.24xlarge):'
\echo '  ACORN-1:  TTI =  8.6s,  Size = 0.93 GB'
\echo '  HNSW:     TTI = 11.3s,  Size = 0.75 GB'
\echo ''

SET maintenance_work_mem = '4GB';

-- Build ACORN-1 index and time it
DROP INDEX IF EXISTS idx_sift_acorn;
DROP INDEX IF EXISTS idx_sift_hnsw_plain;

\echo 'Building ACORN-1 index...'
\timing on
CREATE INDEX idx_sift_acorn ON sift_base
    USING hnsw (embedding vector_l2_ops)
    INCLUDE (tag_id)
    WITH (m = 32, ef_construction = 64);
\timing off

-- Build plain HNSW index and time it
\echo 'Building plain HNSW index...'
\timing on
CREATE INDEX idx_sift_hnsw_plain ON sift_base
    USING hnsw (embedding vector_l2_ops)
    WITH (m = 32, ef_construction = 64);
\timing off

\echo ''
\echo 'Index sizes:'
SELECT
    'ACORN-1 (HNSW + INCLUDE tag_id)' AS method,
    pg_size_pretty(pg_relation_size('idx_sift_acorn')) AS index_size
UNION ALL
SELECT
    'HNSW plain (post-filter baseline)',
    pg_size_pretty(pg_relation_size('idx_sift_hnsw_plain'));

\echo ''
\echo 'Total table + index footprint:'
SELECT pg_size_pretty(pg_total_relation_size('sift_base')) AS total_size;


-- #############################################################################
-- PHASE 6: SUMMARY
-- #############################################################################

\echo ''
\echo '================================================================'
\echo 'SUMMARY: How to interpret your results'
\echo '================================================================'
\echo ''
\echo 'METHOD 1 — ACORN-1 (predicate subgraph traversal):'
\echo '  - Filtering happens INSIDE the HNSW index via GUC'
\echo '  - No WHERE clause in the SQL query'
\echo '  - Should achieve higher recall than post-filter at same ef_search'
\echo '  - Paper: ~1000 distance computations for 0.8 Recall@10'
\echo ''
\echo 'METHOD 2 — HNSW Post-filter (executor-driven):'
\echo '  - Plain HNSW returns candidates; executor filters by tag'
\echo '  - WHERE clause in the SQL query'
\echo '  - Wastes distance computations on non-matching rows'
\echo '  - Paper: ~1838 distance computations for 0.8 Recall@10'
\echo ''
\echo 'METHOD 3 — Pre-filter (brute force):'
\echo '  - No index at all; sequential scan + exact distance sort'
\echo '  - Perfect recall (1.0) but lowest QPS'
\echo '  - Single data point, no ef_search parameter'
\echo ''
\echo 'Expected ordering at the same recall level:'
\echo '  QPS:  ACORN-1 > Post-filter >> Pre-filter'
\echo ''
\echo 'Your absolute QPS will be lower than the paper because:'
\echo '  - Paper: C++ on FAISS, in-memory, 96 vCPUs'
\echo '  - You: PostgreSQL, disk-resident, SQL overhead'
\echo '  - The RELATIVE ordering should match the paper'
\echo '================================================================'
