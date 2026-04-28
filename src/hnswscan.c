#include "postgres.h"

#include "access/genam.h"
#include "access/relscan.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "pgstat.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/relcache.h"
#include "utils/snapmgr.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/*
* Algorithm 5 from paper
*/
/*------------------------------------------*/

// static List *
// GetScanItems(IndexScanDesc scan, Datum value)
// {
// 	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
// 	Relation	index = scan->indexRelation;
// 	HnswSupport *support = &so->support;
// 	List	   *ep;
// 	List	   *w;
// 	int			m;
// 	HnswElement entryPoint;
// 	char	   *base = NULL;
// 	HnswQuery  *q = &so->q;

// 	/* Get m and entry point */
// 	HnswGetMetaPageInfo(index, &m, &entryPoint);

// 	q->value = value;
// 	so->m = m;

// 	if (entryPoint == NULL)
// 		return NIL;

// 	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

// 	for (int lc = entryPoint->level; lc >= 1; lc--)
// 	{
// 		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
// 		ep = w;
// 	}

// 	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples);
// }
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so    = (HnswScanOpaque) scan->opaque;
	Relation       index = scan->indexRelation;
	HnswSupport   *support = &so->support;
	List          *ep;
	List          *w;
	int            m;
	HnswElement    entryPoint;
	char          *base = NULL;
	HnswQuery     *q    = &so->q;

	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value      = value;
	q->has_filter = false;      /* ACORN: default off */
	q->filter_label = HNSW_LABEL_NONE; /* ACORN: invalid label */
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	/* ACORN: engage filtered path when GUC is set */
	if (hnsw_acorn_filter_label >= 0)
	{
		q->has_filter   = true;
		q->filter_label = hnsw_acorn_filter_label;
	}

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));
	/*
	* Upper layers (L >= 1).
	*
	* Paper §5.1, Algorithm 2: ACORN-SEARCH-LAYER runs at EVERY level,
	* beginning from the top entry-point. Paper §6.3.2 describes the
	* upper layers as Stage 1 of the search, where filtering drops
	* down each level until a predicate-matching node is found, then
	* Stage 2 traverses the predicate subgraph at L0.
	*
	* So when filtering is active we route EVERY layer through
	* HnswSearchLayerAcorn (ef = 1 on upper layers, ef = ef_search at
	* L0), matching the paper exactly. When filtering is off, we use
	* the unchanged unfiltered HnswSearchLayer.
	*/
	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m,
								false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	/* Bottom layer (L = 0). */
	if (q->has_filter)
		return HnswSearchLayerAcorn(base, q, ep, hnsw_ef_search, 0, index,
					support, m, false, NULL, &so->v,
					hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF
						? &so->discarded : NULL,
					true, &so->tuples);

	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index,
				support, m, false, NULL, &so->v,
				hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF
					? &so->discarded : NULL,
				true, &so->tuples);
}

/*-----------------------------------------------------*/





/*
* Resume scan at ground level with discarded candidates
*/


/*--------------------------------------------------------------------*/
// static List *
// ResumeScanItems(IndexScanDesc scan)
// {
// 	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
// 	Relation	index = scan->indexRelation;
// 	List	   *ep = NIL;
// 	char	   *base = NULL;
// 	int			batch_size = hnsw_ef_search;

// 	if (pairingheap_is_empty(so->discarded))
// 		return NIL;

// 	/* Get next batch of candidates */
// 	for (int i = 0; i < batch_size; i++)
// 	{
// 		HnswSearchCandidate *sc;

// 		if (pairingheap_is_empty(so->discarded))
// 			break;

// 		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

// 		ep = lappend(ep, sc);
// 	}

// 	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples);
// }
/*---------------------------------------------------------------------------------------------*/
/*
* Resume scan at ground level with discarded candidates
*/
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so    = (HnswScanOpaque) scan->opaque;
	Relation       index = scan->indexRelation;
	List          *ep    = NIL;
	char          *base  = NULL;
	int            batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));
		ep = lappend(ep, sc);
	}

	/*
	* ACORN: so->q.has_filter is set once in GetScanItems() and
	* GetScanItems() and remain valid for the whole scan lifetime
	* (so->tmpCtx is not reset between iterations).
	* Route to the ACORN variant when filtering is active, exactly
	* as GetScanItems() does.
	*
	* Note: initVisited=false here — the visited hash is reused
	* across iterations. This matches the original call.
	*/
	if (so->q.has_filter)
		return HnswSearchLayerAcorn(base, &so->q, ep, batch_size, 0,
					index, &so->support, so->m, false, NULL,
					&so->v, &so->discarded, false, &so->tuples);

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0,
				index, &so->support, so->m, false, NULL,
					&so->v, &so->discarded, false, &so->tuples);
}
/*-----------------------------------------------------------------------------------------------*/
/*
* Get scan value
*/
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
* Show memory usage
*/
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
* Prepare for an index scan
*/
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	* Use a lower max allocation size than default to allow scanning more
	* tuples for iterative search before exceeding work_mem
	*/
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									"Hnsw scan temporary context",
									0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

	return scan;
}

/*
* Start or restart an index scan
*/
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);
	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
* Fetch the next tuple in the given scan
*/
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	* Index can be used to scan backward, but Postgres doesn't support
	* backward scan on operators
	*/
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		* Get a shared lock. This allows vacuum to ensure no in-flight scans
		* before marking tuples as deleted.
		*/
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;
		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				* Locking ensures when neighbors are read, the elements they
				* reference will not be deleted (and replaced) during the
				* iteration.
				*
				* Elements loaded into memory on previous iterations may have
				* been deleted (and replaced), so when reading neighbors, the
				* element version must be checked.
				*/
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}
		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		heaptid = NULL;
// /* DEBUG: trace what's being returned */
// if (so->q.has_filter)
//     elog(NOTICE, "gettuple: blk=%u off=%u len=%d labels=[%d,%d] filter=%d",
//          element->blkno, element->offno,
//          element->heaptidsLength,
//          element->heaptidsLength > 0 ? element->labels[0] : -1,
//          element->heaptidsLength > 1 ? element->labels[1] : -1,
//          so->q.filter_label);
		/* ACORN: Drain non-matching TIDs from the element */
		while (element->heaptidsLength > 0)
		{
			int idx = element->heaptidsLength - 1;

			if (!so->q.has_filter || element->labels[idx] == so->q.filter_label)
			{
				heaptid = &element->heaptids[idx];
				element->heaptidsLength--;
				break;
			}

			/* Discard non-matching TID and check the next one */
			element->heaptidsLength--;
		}

		/* Move to next element if no valid heap TIDs remain */
		if (heaptid == NULL)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
* End a scan and release resources
*/
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}