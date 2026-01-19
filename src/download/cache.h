/*
 * cache.h - Model cache management
 *
 * Manages local cache of downloaded models with LRU eviction.
 */

#ifndef CACHE_H
#define CACHE_H

/* Cache entry */
typedef struct CacheEntry CacheEntry;
struct CacheEntry {
    char *repo_id;      /* HuggingFace repo ID */
    char *filename;     /* File name */
    char *localpath;    /* Full local path */
    vlong size;         /* File size in bytes */
    vlong atime;        /* Last access time (nsec) */
    CacheEntry *next;
    CacheEntry *prev;
};

/* Cache state */
typedef struct ModelCache ModelCache;
struct ModelCache {
    char *cache_dir;        /* Cache directory path */
    CacheEntry *head;       /* Most recently used */
    CacheEntry *tail;       /* Least recently used */
    vlong total_size;       /* Total cache size */
    vlong max_size;         /* Maximum cache size (0 = unlimited) */
    int num_entries;
};

/* Initialize cache */
void cache_init(ModelCache *c, char *cache_dir, vlong max_size);

/* Close cache and free resources */
void cache_close(ModelCache *c);

/* Scan cache directory and build index */
int cache_scan(ModelCache *c);

/* Look up a file in cache, returns path if found, nil otherwise */
char *cache_lookup(ModelCache *c, char *repo_id, char *filename);

/* Add a file to cache */
int cache_add(ModelCache *c, char *repo_id, char *filename,
              char *localpath, vlong size);

/* Remove a file from cache */
int cache_remove(ModelCache *c, char *repo_id, char *filename);

/* Evict files to make room for new_size bytes */
int cache_evict(ModelCache *c, vlong new_size);

/* Get cache statistics */
void cache_stats(ModelCache *c, vlong *total_size, int *num_entries);

/* Mark entry as recently used */
void cache_touch(ModelCache *c, CacheEntry *e);

/* Find entry by repo_id and filename */
CacheEntry *cache_find(ModelCache *c, char *repo_id, char *filename);

/* List all cached files for a repo */
CacheEntry *cache_list_repo(ModelCache *c, char *repo_id);

/* Clear all cache entries */
void cache_clear(ModelCache *c);

#endif /* CACHE_H */
