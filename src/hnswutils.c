#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "common/hashfn.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "nodes/pg_list.h"
#include "port/atomics.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/memdebug.h"
#include "utils/rel.h"
#include "vector.h"
// NEWLY ADDED: added*/
/* ACORN: label resolution at insert time */
#include "access/heapam.h"     /* heap_fetch */
#include "access/tableam.h"
#include "utils/snapmgr.h"
#include "executor/tuptable.h"
// NEWLY ADDED: added*/
#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Init support functions
 */
void
HnswInitSupport(HnswSupport * support, Relation index)
{
	support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
}

/*
 * Normalize value
 */
Datum
HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
HnswCheckNorm(HnswSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator * allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * allocator)
{
	int			level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *) HnswAlloc(allocator, sizeof(HnswNeighborArrayPtr) * (level + 1));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
// NEWLY ADDED: ---*/
// HnswElement
// HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
// {
// 	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

// 	int			level = (int) (-log(RandomDouble()) * ml);

// 	/* Cap level */
// 	if (level > maxLevel)
// 		level = maxLevel;

// 	element->heaptidsLength = 0;
// 	HnswAddHeapTid(element, heaptid);

// 	element->level = level;
// 	element->deleted = 0;
// 	/* Start at one to make it easier to find issues */
// 	element->version = 1;

// 	HnswInitNeighbors(base, element, m, allocator);

// 	HnswPtrStore(base, element->value, (char *) NULL);

// 	return element;
// }
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
{
	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	int			level = (int) (-log(RandomDouble()) * ml);

	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	/* Seed with HNSW_LABEL_NONE; caller overwrites via HnswAddHeapTid. */
	HnswAddHeapTid(element, heaptid, HNSW_LABEL_NONE);

	element->level = level;
	element->deleted = 0;
	element->version = 1;

	HnswInitNeighbors(base, element, m, allocator);
	HnswPtrStore(base, element->value, (char *) NULL);

	return element;
}
// END ADD: ---*/
/*
 * Add a heap TID to an element
 */
// NEWLY ADDED:--*/
// void
// HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
// {
// 	element->heaptids[element->heaptidsLength++] = *heaptid;
// }
void
HnswAddHeapTid(HnswElement element, ItemPointer heaptid, int32 label)
{
	element->heaptids[element->heaptidsLength] = *heaptid;
	element->labels[element->heaptidsLength] = label;
	element->heaptidsLength++;
}
// END ADD:--*/

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc(sizeof(HnswElementData));
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *) NULL);
	HnswPtrStore(base, element->value, (char *) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void
HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Form index value
 */
bool
HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!HnswCheckNorm(support, value))
			return false;

		value = HnswNormValue(typeInfo, support->collation, value);
	}

	*out = value;

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
// NEWLY ADDED:---------*/
// void
// HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
// {
// 	Pointer		valuePtr = HnswPtrAccess(base, element->value);

// 	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
// 	etup->level = element->level;
// 	etup->deleted = 0;
// 	etup->version = element->version;
// 	for (int i = 0; i < HNSW_HEAPTIDS; i++)
// 	{
// 		if (i < element->heaptidsLength)
// 			etup->heaptids[i] = element->heaptids[i];
// 		else
// 			ItemPointerSetInvalid(&etup->heaptids[i]);
// 	}
// 	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
// }
void
HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer		valuePtr = HnswPtrAccess(base, element->value);

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
		{
			etup->heaptids[i] = element->heaptids[i];
			etup->labels[i]   = element->labels[i];
		}
		else
		{
			ItemPointerSetInvalid(&etup->heaptids[i]);
			etup->labels[i] = HNSW_LABEL_NONE;
		}
	}
	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
}
// END ADD: -------------*/
/*
 * Set neighbor tuple
 */
void
HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m)
{
	int			idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int			lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
// NEWLY ADDED:---------*/
// void
// HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
// {
// 	element->level = etup->level;
// 	element->deleted = etup->deleted;
// 	element->version = etup->version;
// 	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
// 	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
// 	element->heaptidsLength = 0;

// 	if (loadHeaptids)
// 	{
// 		for (int i = 0; i < HNSW_HEAPTIDS; i++)
// 		{
// 			/* Can stop at first invalid */
// 			if (!ItemPointerIsValid(&etup->heaptids[i]))
// 				break;

// 			HnswAddHeapTid(element, &etup->heaptids[i]);
// 		}
// 	}

// 	if (loadVec)
// 	{
// 		char	   *base = NULL;
// 		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

// 		HnswPtrStore(base, element->value, (char *) DatumGetPointer(value));
// 	}
// }
void
HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			/* Pass the stored label in lockstep with the heap TID. */
			HnswAddHeapTid(element, &etup->heaptids[i], etup->labels[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, (char *) DatumGetPointer(value));
	}
}
// NEWLY ADDED: -------------*/

/*
 * Calculate the distance between values
 */
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport * support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

/*
 * Load an element and optionally get its distance from q
 */
static void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element)
{
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;

	/* Read vector */
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

	Assert(HnswIsElementTuple(etup));

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(q->value) == NULL)
			*distance = 0;
		else
			*distance = HnswGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
	{
		if (*element == NULL)
			*element = HnswInitElementFromBlock(blkno, offno);

		HnswLoadElementFromTuple(*element, etup, true, loadVec);
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance)
{
	HnswLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element);
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, HnswElement element, HnswQuery * q, HnswSupport * support)
{
	Datum		value = HnswGetValue(base, element);

	return HnswGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static HnswSearchCandidate *
HnswInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
HnswSearchCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		HnswLoadElement(entryPoint, &distance, q, index, support, loadVec, NULL);

	return HnswInitSearchCandidate(base, entryPoint, distance);
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(c_node, a)->distance < HnswGetSearchCandidateConst(c_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(c_node, a)->distance > HnswGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m)
{
	if (!inMemory)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, HnswElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) HnswPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

/*
 * Load unvisited neighbors from memory
 */
static void
HnswLoadUnvisitedFromMemory(char *base, HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, HnswNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs
 */
bool
HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc)
{
	Buffer		buf;
	Page		page;
	HnswNeighborTuple ntup;
	int			start;

	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = (element->level - lc) * m;
	memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return true;
}

/*
 * Load unvisited neighbors from disk
 */
static void
HnswLoadUnvisitedFromDisk(HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, Relation index, int m, int lm, int lc)
{
	ItemPointerData indextids[HNSW_MAX_M * 2];

	*unvisitedLength = 0;

	if (!HnswLoadNeighborTids(element, indextids, index, m, lm, lc))
		return;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		bool		found;

		if (!ItemPointerIsValid(indextid))
			break;

		tidhash_insert(v->tids, *indextid, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].indextid = *indextid;
	}
}

/*
 * Algorithm 2 from paper
 */
List *
HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples)
{
	List	   *w = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int			wlen = 0;
	visited_hash vh;
	ListCell   *lc2;
	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm = HnswGetLayerM(m, lc);
	HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	int			unvisitedLength;
	bool		inMemory = index == NULL;

	if (v == NULL)
	{
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	/* Create local memory for neighborhood if needed */
	if (inMemory)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
				(*tuples)++;
		}

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		HnswSearchCandidate *f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;

		if (c->distance > f->distance)
			break;

		cElement = HnswPtrAccess(base, c->element);

		if (inMemory)
			HnswLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		else
			HnswLoadUnvisitedFromDisk(cElement, unvisited, &unvisitedLength, v, index, m, lm, lc);

		/* OK to count elements instead of tuples */
		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		for (int i = 0; i < unvisitedLength; i++)
		{
			HnswElement eElement;
			HnswSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;

			f = HnswGetSearchCandidate(w_node, pairingheap_first(W));

			if (inMemory)
			{
				eElement = unvisited[i].element;
				eDistance = GetElementDistance(base, eElement, q, support);
			}
			else
			{
				ItemPointer indextid = &unvisited[i].indextid;
				BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);

				/* Avoid any allocations if not adding */
				eElement = NULL;
				HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, alwaysAdd || discarded != NULL ? NULL : &f->distance, &eElement);

				if (eElement == NULL)
					continue;
			}
			if (!(eDistance < f->distance || alwaysAdd))
			{
				if (discarded != NULL)
				{
					/* Create a new candidate */
					e = HnswInitSearchCandidate(base, eElement, eDistance);
					pairingheap_add(*discarded, &e->w_node);
				}

				continue;
			}

			/* Make robust to issues */
			if (eElement->level < lc)
				continue;
			/* Create a new candidate */
			e = HnswInitSearchCandidate(base, eElement, eDistance);
			pairingheap_add(C, &e->c_node);
			pairingheap_add(W, &e->w_node);

			/*
			 * Do not count elements being deleted towards ef when vacuuming.
			 * It would be ideal to do this for inserts as well, but this
			 * could affect insert performance.
			 */
			if (CountElement(skipElement, eElement))
			{
				wlen++;

				/* No need to decrement wlen */
				if (wlen > ef)
				{
					HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

					if (discarded != NULL)
						pairingheap_add(*discarded, &d->w_node);
				}
			}
		}
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

/*
 * Compare candidate distances with pointer tie-breaker
 */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;

	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;

	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, HnswCandidate * e, List *r, HnswSupport * support)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	Datum		eValue = HnswGetValue(base, eElement);
	ListCell   *lc2;

	foreach(lc2, r)
	{
		HnswCandidate *ri = lfirst(lc2);
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		Datum		riValue = HnswGetValue(base, riElement);
		float		distance = HnswGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, HnswSupport * support, bool *closerSet, HnswCandidate * newCandidate, HnswCandidate * *pruned, bool sortCandidates)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	HnswCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;
	bool		mustCalculate = !(*closerSet);
	List	   *added = NIL;
	bool		removedAny = false;

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(HnswCandidate *) * list_length(w));

	/* Ensure order of candidates is deterministic for closer caching */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		HnswCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, support);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, support);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, support);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support)
{
	HnswCandidate newHc;

	HnswPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		HnswCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (HnswPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *) lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	HnswQuery	q;
	HnswElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

/* BEFORE: */
    // q.value = HnswGetValue(base, element);

/* AFTER: */
    q.value      = HnswGetValue(base, element);
    q.has_filter = false;     /* ACORN: insert path never filters */
    q.filter_label = -1;
/*-----------------------------------------------*/
	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		List	   *neighbors;
		List	   *lw = NIL;
		ListCell   *lc2;

		w = HnswSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);

		/* Convert search candidates to candidates */
		foreach(lc2, w)
		{
			HnswSearchCandidate *sc = lfirst(lc2);
			HnswCandidate *hc = palloc(sizeof(HnswCandidate));

			hc->element = sc->element;
			hc->distance = sc->distance;

			lw = lappend(lw, hc);
		}

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (!inMemory)
			lw = RemoveElements(base, lw, skipElement);

		/*
		 * Candidates are sorted, but not deterministically. Could set
		 * sortCandidates to true for in-memory builds to enable closer
		 * caching, but there does not seem to be a difference in performance.
		 */
		neighbors = SelectNeighbors(base, lw, lm, support, &HnswGetNeighbors(base, element, lc)->closerSet, NULL, NULL, false);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

// NEWLY ADDED:*/
/*
 * AcornElementMatches
 *
 * An element can carry up to HNSW_HEAPTIDS heap TIDs (same-vector dedup,
 * non-HOT update chains). Each has its own label in the parallel array.
 * The element matches the filter if ANY live slot has the target label.
 *
 * O(heaptidsLength) with HNSW_HEAPTIDS = 10. Call it constant-time.
 */
bool
AcornElementMatches(HnswElement element, int32 filter_label)
{
	for (int i = 0; i < element->heaptidsLength; i++)
	{
		if (element->labels[i] == filter_label)
			return true;
	}
	return false;
}
/*
 * HnswLoadUnvisitedAcorn — ACORN-1 two-hop expansion, paper §5.3.
 *
 * Walks hop-1 and hop-2 neighbors of the current node. Admits a
 * candidate to the unvisited pool iff its stored label matches the
 * filter. Non-matching hop-1 nodes are still traversed for hop-2 —
 * that's the whole point of ACORN: go THROUGH non-matching nodes
 * to reach matching ones.
 *
 * Labels are read directly from the element tuple we had to fetch
 * anyway (HnswLoadElement reads the tuple containing heaptids and
 * labels). No auxiliary hash. No heap access.
 *
 * Invariants:
 *   - Visited hash keyed by INDEX TID (graph dedup).
 *   - Label check = AcornElementMatches(elem, filter).
 *   - Output capped at lm.
 */
static void
HnswLoadUnvisitedAcorn(HnswElement element, HnswUnvisited *unvisited,
                       int *unvisitedLength, visited_hash *v,
                       Relation index, int m, int lm, int lc,
                       int32 filter_label) /* ADDED PARAMETER */
{
    ItemPointerData hop1Tids[HNSW_MAX_M * 2];
    visited_hash local_v;
	int max_exploration_size = lm + (lm * lm);
    *unvisitedLength = 0;
    if (!HnswLoadNeighborTids(element, hop1Tids, index, m, lm, lc))
        return;

    /*
     * Local dedup hash — scoped to this single expansion call.
     * Prevents the same node from being returned twice when it
     * appears as a hop-2 neighbor of multiple hop-1 bridges.
     * Destroyed at the end of this function.
     */

    local_v.tids = tidhash_create(CurrentMemoryContext, max_exploration_size, NULL);

    for (int i = 0; i < lm; i++)
    {
        ItemPointerData hop1Tid = hop1Tids[i];
        HnswElement     hop1Elem;
        bool            hop1LocallySeen;
        bool            hop1Matches;

        if (!ItemPointerIsValid(&hop1Tid))
            break;

        /*
         * READ-ONLY check on the global visited hash.
         * If a prior expansion already fully processed this node
         * (either as a central node or as a bridge), skip it entirely.
         */
        if (tidhash_lookup(v->tids, hop1Tid) != NULL)
            continue;

        /* Local dedup — skip if already seen in this expansion */
        tidhash_insert(local_v.tids, hop1Tid, &hop1LocallySeen);
        if (hop1LocallySeen)
            continue;

        /* Load element to check label and get neighbor page info */
        hop1Elem = HnswInitElementFromBlock(ItemPointerGetBlockNumber(&hop1Tid),
                                            ItemPointerGetOffsetNumber(&hop1Tid));
        HnswLoadElement(hop1Elem, NULL, NULL, index, NULL, false, NULL);

        /* Check if hop-1 matches the predicate */
        hop1Matches = AcornElementMatches(hop1Elem, filter_label);

        /* CRITICAL FIX: If matching, add to the output pool WITH safety cap */
        if (hop1Matches){
            unvisited[(*unvisitedLength)++].indextid = hop1Tid;
			 if (*unvisitedLength >= lm)
			 {
				 pfree(hop1Elem);
				 break;
			 }
		}

        /*
         * Hop-2: walk THROUGH hop1Elem regardless of whether it
         * matched the filter. Non-matching nodes are bridges.
         */
        {
            ItemPointerData hop2Tids[HNSW_MAX_M * 2];

            if (HnswLoadNeighborTids(hop1Elem, hop2Tids, index, m, lm, lc))
            {
                for (int j = 0; j < lm; j++)
                {
                    ItemPointerData hop2Tid = hop2Tids[j];
                    bool            hop2LocallySeen;
                    HnswElement     hop2Elem;

                    if (!ItemPointerIsValid(&hop2Tid))
                        break;

                    /* READ-ONLY check on global visited hash */
                    if (tidhash_lookup(v->tids, hop2Tid) != NULL)
                        continue;

                    /* Local dedup */
                    tidhash_insert(local_v.tids, hop2Tid, &hop2LocallySeen);
                    if (hop2LocallySeen)
                        continue;

                    /* Load element for label check */
                    hop2Elem = HnswInitElementFromBlock(
                                    ItemPointerGetBlockNumber(&hop2Tid),
                                    ItemPointerGetOffsetNumber(&hop2Tid));
                    HnswLoadElement(hop2Elem, NULL, NULL, index, NULL, false, NULL);

                    /* CRITICAL FIX: Add to the output pool WITH safety cap */
                    if (AcornElementMatches(hop2Elem, filter_label)){
                        unvisited[(*unvisitedLength)++].indextid = hop2Tid;
						if(*unvisitedLength >= lm)
						{
							pfree(hop2Elem);
							break;
						}
					}
                    pfree(hop2Elem);
                }
            }
			if(*unvisitedLength >= lm){
				pfree(hop1Elem);
				break;
			}
        }
        pfree(hop1Elem);
    }

    tidhash_destroy(local_v.tids);
}
/*
 * HnswSearchLayerAcorn
 *
 * Near-identical copy of HnswSearchLayer (Algorithm 2).
 * THE ONE CONTROLLED CHANGE: in the disk path, calls
 * HnswLoadUnvisitedAcorn instead of HnswLoadUnvisitedFromDisk.
 *
 * Lives in hnswutils.c so it can call the static helper functions
 * (InitVisited, AddToVisited, CountElement, the comparators,
 * HnswLoadElementImpl) without any API surface changes.
 *
 * The inMemory path (index build) falls through to the standard
 * unfiltered expansion — ACORN filtering is query-time only.
 *
 * HnswSearchLayer is never modified.
 */

List *
HnswSearchLayerAcorn(char *base, HnswQuery *q, List *ep, int ef, int lc,
                     Relation index, HnswSupport *support, int m,
                     bool inserting, HnswElement skipElement,
                     visited_hash *v, pairingheap **discarded,
                     bool initVisited, int64 *tuples)
{
    List       *w    = NIL;
    pairingheap *C   = pairingheap_allocate(CompareNearestCandidates, NULL);
    pairingheap *W   = pairingheap_allocate(CompareFurthestCandidates, NULL);
    int         wlen = 0;
    visited_hash vh;
    ListCell   *lc2;
    int         lm   = HnswGetLayerM(m, lc);

    /*
     * Allocate enough space for the full uncapped 2-hop pool.
     * HnswLoadUnvisitedAcorn may return up to lm + lm*lm candidates.
     * The C/W heaps handle distance-aware truncation to ef.
     */
    HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
    int         unvisitedLength;
    bool        inMemory      = index == NULL;

    /* Identical to HnswSearchLayer */
    if (v == NULL)
    {
        v           = &vh;
        initVisited = true;
    }

    if (initVisited)
    {
        InitVisited(base, v, inMemory, ef, m);

        if (discarded != NULL)
            *discarded = pairingheap_allocate(CompareNearestDiscardedCandidates,
                                              NULL);
    }

    /* Add entry points to v, C, and W — identical to HnswSearchLayer */
    foreach(lc2, ep)
    {
        HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
        bool found;

        if (initVisited)
        {
            AddToVisited(base, v, sc->element, inMemory, &found);

            if (tuples != NULL)
                (*tuples)++;
        }

        pairingheap_add(C, &sc->c_node);
        pairingheap_add(W, &sc->w_node);

        if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
            wlen++;
    }

    /* Main search loop */
    while (!pairingheap_is_empty(C))
    {
        HnswSearchCandidate *c = HnswGetSearchCandidate(c_node,
                                     pairingheap_remove_first(C));
        HnswSearchCandidate *f = HnswGetSearchCandidate(w_node,
                                     pairingheap_first(W));
        HnswElement cElement;

        if (c->distance > f->distance)
            break;

        cElement = HnswPtrAccess(base, c->element);

        if (inMemory)
        {
            /*
             * In-memory path: index build only, never at query time.
             * ACORN filtering does not apply during construction.
             */
            HnswNeighborArray  *localNeighborhood;
            Size                neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);

            localNeighborhood = palloc(neighborhoodSize);
            HnswLoadUnvisitedFromMemory(base, cElement, unvisited,
                                        &unvisitedLength, v, lc,
                                        localNeighborhood, neighborhoodSize);
            pfree(localNeighborhood);
        }
        else
        {
            /*
             * Disk path — 2-hop expansion with predicate filter.
             * Returns the full uncapped pool of matching candidates.
             * CRITICAL FIX: Pass maxPoolSize to prevent buffer overflow.
             */
            HnswLoadUnvisitedAcorn(cElement, unvisited, &unvisitedLength,
                                   v, index, m, lm, lc,
                                   q->filter_label/* ADDED ARGUMENT */);
        }

        if (tuples != NULL)
            (*tuples) += unvisitedLength;

        /* Evaluate each candidate from the pool */
        for (int i = 0; i < unvisitedLength; i++)
        {
            HnswElement          eElement;
            HnswSearchCandidate *e;
            double               eDistance;
            bool                 alwaysAdd = wlen < ef;

            f = HnswGetSearchCandidate(w_node, pairingheap_first(W));

            /*
             * DELAYED COMMIT: mark the candidate as globally visited
             * right before evaluation. HnswLoadUnvisitedAcorn uses
             * only a local hash, so matching candidates have not yet
             * been added to v->tids. This is where they get committed.
             * Duplicates (same node returned by overlapping expansions)
             * are caught here and skipped.
             */
            if (!inMemory)
            {
                ItemPointer indextid = &unvisited[i].indextid;
                bool        alreadyVisited;

                tidhash_insert(v->tids, *indextid, &alreadyVisited);
                if (alreadyVisited)
                    continue;
            }

            if (inMemory)
            {
                eElement  = unvisited[i].element;
                eDistance = GetElementDistance(base, eElement, q, support);
            }
            else
            {
                ItemPointer  indextid = &unvisited[i].indextid;
                BlockNumber  blkno    = ItemPointerGetBlockNumber(indextid);
                OffsetNumber offno    = ItemPointerGetOffsetNumber(indextid);

                eElement = NULL;
                HnswLoadElementImpl(blkno, offno, &eDistance, q, index,
                                    support, inserting,
                                    alwaysAdd || discarded != NULL
                                        ? NULL : &f->distance,
                                    &eElement);

                if (eElement == NULL)
                    continue;
            }

            if (!(eDistance < f->distance || alwaysAdd))
            {
                if (discarded != NULL)
                {
                    e = HnswInitSearchCandidate(base, eElement, eDistance);
                    pairingheap_add(*discarded, &e->w_node);
                }
                continue;
            }

            if (eElement->level < lc)
                continue;

            e = HnswInitSearchCandidate(base, eElement, eDistance);
            pairingheap_add(C, &e->c_node);
            pairingheap_add(W, &e->w_node);

            if (CountElement(skipElement, eElement))
            {
                wlen++;

                if (wlen > ef)
                {
                    HnswSearchCandidate *d = HnswGetSearchCandidate(w_node,
                                                pairingheap_remove_first(W));
                    if (discarded != NULL)
                        pairingheap_add(*discarded, &d->w_node);
                }
            }
        }
    }

    /* Drain W into result list — identical to HnswSearchLayer */
    while (!pairingheap_is_empty(W))
    {
        HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node,
                                      pairingheap_remove_first(W));
        w = lappend(w, sc);
    }

    pfree(unvisited);
    return w;
}
// NEWLY ADDED:*/

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *) v;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ)));
}

/*
 * Get type info
 */
const		HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.normalize = l2_normalize,
			.checkValue = NULL
		};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum
hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum
hnsw_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.normalize = NULL,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum
hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue
	};

	PG_RETURN_POINTER(&typeInfo);
}