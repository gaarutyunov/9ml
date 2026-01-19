/*
 * pool.c - Model pool implementation
 *
 * LRU eviction with reference counting for safe multi-model management.
 *
 * This file can be included after model.c (which provides the headers
 * and model_load/free/memory functions), or compiled standalone.
 */

/* Include required Plan 9 headers only if not already included.
 * When included after model.c, these will already be present. */
#ifndef nil
#include <u.h>
#include <libc.h>
#include <thread.h>
#endif

#include "pool/pool.h"

void
pool_init(ModelPool *p, int max_models, uvlong max_memory)
{
	memset(p, 0, sizeof(*p));
	p->max_models = max_models;
	p->max_memory = max_memory;
}

static void
entry_free(PoolEntry *e)
{
	if (e == nil)
		return;
	if (e->transformer && e->tokenizer)
		model_free(e->transformer, e->tokenizer);
	free(e->name);
	free(e->modelpath);
	free(e->tokenizerpath);
	free(e);
}

void
pool_free(ModelPool *p)
{
	PoolEntry *e, *next;

	qlock(&p->lk);
	for (e = p->head; e != nil; e = next) {
		next = e->next;
		entry_free(e);
	}
	p->head = nil;
	p->tail = nil;
	p->count = 0;
	p->total_memory = 0;
	qunlock(&p->lk);
}

/* Move entry to head of LRU list (most recently used) */
static void
lru_touch(ModelPool *p, PoolEntry *e)
{
	if (e == p->head)
		return;  /* Already at head */

	/* Remove from current position */
	if (e->prev)
		e->prev->next = e->next;
	if (e->next)
		e->next->prev = e->prev;
	if (e == p->tail)
		p->tail = e->prev;

	/* Insert at head */
	e->prev = nil;
	e->next = p->head;
	if (p->head)
		p->head->prev = e;
	p->head = e;
	if (p->tail == nil)
		p->tail = e;

	e->lastuse = nsec();
}

/* Remove entry from LRU list */
static void
lru_remove(ModelPool *p, PoolEntry *e)
{
	if (e->prev)
		e->prev->next = e->next;
	else
		p->head = e->next;

	if (e->next)
		e->next->prev = e->prev;
	else
		p->tail = e->prev;

	e->prev = nil;
	e->next = nil;
}

/* Find entry by name (caller must hold lock) */
static PoolEntry *
find_entry(ModelPool *p, char *name)
{
	PoolEntry *e;

	for (e = p->head; e != nil; e = e->next) {
		if (strcmp(e->name, name) == 0)
			return e;
	}
	return nil;
}

uvlong
pool_evict_lru(ModelPool *p, uvlong bytes_needed)
{
	PoolEntry *e, *prev;
	uvlong freed = 0;

	/* Start from tail (least recently used) */
	e = p->tail;
	while (e != nil && p->total_memory + bytes_needed > p->max_memory) {
		prev = e->prev;

		/* Only evict if not in use */
		if (e->refcount == 0) {
			freed += e->memory;
			p->total_memory -= e->memory;
			p->count--;
			lru_remove(p, e);
			entry_free(e);
		}

		e = prev;
	}

	return freed;
}

PoolEntry *
pool_load(ModelPool *p, char *name, char *modelpath, char *tokenizerpath)
{
	PoolEntry *e;
	uvlong mem;

	qlock(&p->lk);

	/* Check if already loaded */
	e = find_entry(p, name);
	if (e != nil) {
		e->refcount++;
		lru_touch(p, e);
		qunlock(&p->lk);
		return e;
	}

	/* Allocate new entry */
	e = mallocz(sizeof(PoolEntry), 1);
	if (e == nil) {
		qunlock(&p->lk);
		return nil;
	}

	e->name = strdup(name);
	e->modelpath = strdup(modelpath);
	e->tokenizerpath = strdup(tokenizerpath);
	if (e->name == nil || e->modelpath == nil || e->tokenizerpath == nil) {
		entry_free(e);
		qunlock(&p->lk);
		return nil;
	}

	/* Allocate transformer and tokenizer structs */
	e->transformer = mallocz(sizeof(Transformer), 1);
	e->tokenizer = mallocz(sizeof(Tokenizer), 1);
	if (e->transformer == nil || e->tokenizer == nil) {
		entry_free(e);
		qunlock(&p->lk);
		return nil;
	}

	/* Estimate memory needed and evict if necessary */
	/* TODO: Better estimation based on model config */
	mem = 100 * 1024 * 1024;  /* 100MB initial estimate */
	if (p->total_memory + mem > p->max_memory) {
		pool_evict_lru(p, mem);
	}

	/* Check model count limit */
	while (p->count >= p->max_models) {
		PoolEntry *victim = p->tail;
		while (victim && victim->refcount > 0)
			victim = victim->prev;
		if (victim == nil) {
			/* All models in use, can't evict */
			entry_free(e);
			qunlock(&p->lk);
			return nil;
		}
		p->total_memory -= victim->memory;
		p->count--;
		lru_remove(p, victim);
		entry_free(victim);
	}

	/* Load the model (this may block for a while) */
	qunlock(&p->lk);
	if (model_load(e->transformer, e->tokenizer, modelpath, tokenizerpath) != 0) {
		entry_free(e);
		return nil;
	}
	qlock(&p->lk);

	/* Get actual memory usage */
	e->memory = model_memory(e->transformer);
	e->refcount = 1;
	e->lastuse = nsec();

	/* Add to LRU list at head */
	e->next = p->head;
	e->prev = nil;
	if (p->head)
		p->head->prev = e;
	p->head = e;
	if (p->tail == nil)
		p->tail = e;

	p->count++;
	p->total_memory += e->memory;

	qunlock(&p->lk);
	return e;
}

PoolEntry *
pool_get(ModelPool *p, char *name)
{
	PoolEntry *e;

	qlock(&p->lk);
	e = find_entry(p, name);
	if (e != nil) {
		e->refcount++;
		lru_touch(p, e);
	}
	qunlock(&p->lk);
	return e;
}

void
pool_release(ModelPool *p, PoolEntry *e)
{
	if (e == nil)
		return;

	qlock(&p->lk);
	if (e->refcount > 0)
		e->refcount--;
	qunlock(&p->lk);
}

int
pool_unload(ModelPool *p, char *name)
{
	PoolEntry *e;
	int ret = -1;

	qlock(&p->lk);
	e = find_entry(p, name);
	if (e != nil && e->refcount == 0) {
		p->total_memory -= e->memory;
		p->count--;
		lru_remove(p, e);
		entry_free(e);
		ret = 0;
	}
	qunlock(&p->lk);
	return ret;
}

int
pool_count(ModelPool *p)
{
	int n;
	qlock(&p->lk);
	n = p->count;
	qunlock(&p->lk);
	return n;
}

uvlong
pool_memory(ModelPool *p)
{
	uvlong m;
	qlock(&p->lk);
	m = p->total_memory;
	qunlock(&p->lk);
	return m;
}

char *
pool_list(ModelPool *p)
{
	PoolEntry *e;
	int len, cap;
	char *buf;

	qlock(&p->lk);

	if (p->count == 0) {
		qunlock(&p->lk);
		return strdup("");
	}

	cap = 256;
	buf = malloc(cap);
	if (buf == nil) {
		qunlock(&p->lk);
		return nil;
	}
	len = 0;
	buf[0] = '\0';

	for (e = p->head; e != nil; e = e->next) {
		int namelen = strlen(e->name);
		if (len + namelen + 2 >= cap) {
			cap *= 2;
			char *newbuf = realloc(buf, cap);
			if (newbuf == nil) {
				free(buf);
				qunlock(&p->lk);
				return nil;
			}
			buf = newbuf;
		}
		if (len > 0) {
			buf[len++] = ',';
		}
		strcpy(buf + len, e->name);
		len += namelen;
	}

	qunlock(&p->lk);
	return buf;
}
