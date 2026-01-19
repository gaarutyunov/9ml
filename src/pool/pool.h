/*
 * pool.h - Model pool management for llmfs
 *
 * Manages multiple loaded models with LRU eviction.
 * Thread-safe with reference counting for in-use models.
 *
 * Note: This header requires <u.h>, <libc.h>, <thread.h> to be included first.
 */

#ifndef POOL_H
#define POOL_H

/* Forward declarations */
typedef struct PoolEntry PoolEntry;
typedef struct ModelPool ModelPool;

/* Pool entry - one loaded model */
struct PoolEntry {
	char *name;              /* Model identifier (e.g., "llama2-7b") */
	char *modelpath;         /* Path to model file */
	char *tokenizerpath;     /* Path to tokenizer file */
	void *transformer;       /* Loaded Transformer* */
	void *tokenizer;         /* Loaded Tokenizer* */
	uvlong memory;           /* Memory used by this model */
	int refcount;            /* Number of sessions using this model */
	vlong lastuse;           /* nsec() of last access (for LRU) */
	PoolEntry *next;         /* Next in LRU list (more recent) */
	PoolEntry *prev;         /* Previous in LRU list (less recent) */
};

/* Model pool - manages multiple models */
struct ModelPool {
	PoolEntry *head;         /* Most recently used */
	PoolEntry *tail;         /* Least recently used */
	int count;               /* Number of loaded models */
	int max_models;          /* Maximum models to keep loaded */
	uvlong total_memory;     /* Total memory used by all models */
	uvlong max_memory;       /* Maximum memory to use */
	QLock lk;                /* Pool lock */
};

/* Initialize pool with limits */
void pool_init(ModelPool *p, int max_models, uvlong max_memory);

/* Free pool and all entries */
void pool_free(ModelPool *p);

/* Load a model (may evict LRU entries to make room)
 * Returns entry, or nil on failure */
PoolEntry *pool_load(ModelPool *p, char *name, char *modelpath, char *tokenizerpath);

/* Get a model by name (increments refcount, moves to head)
 * Returns entry with refcount incremented, or nil if not found */
PoolEntry *pool_get(ModelPool *p, char *name);

/* Release a model (decrements refcount)
 * Must be called when session stops using the model */
void pool_release(ModelPool *p, PoolEntry *e);

/* Unload a specific model by name (fails if refcount > 0)
 * Returns 0 on success, -1 if not found or in use */
int pool_unload(ModelPool *p, char *name);

/* Evict LRU entries to free at least bytes_needed
 * Only evicts entries with refcount == 0
 * Returns bytes freed */
uvlong pool_evict_lru(ModelPool *p, uvlong bytes_needed);

/* Get pool statistics */
int pool_count(ModelPool *p);
uvlong pool_memory(ModelPool *p);

/* List all models (returns comma-separated names, caller frees) */
char *pool_list(ModelPool *p);

#endif /* POOL_H */
