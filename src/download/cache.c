/*
 * cache.c - Model cache management
 *
 * LRU cache for downloaded model files.
 */

#include <u.h>
#include <libc.h>

#include "cache.h"

void
cache_init(ModelCache *c, char *cache_dir, vlong max_size)
{
    memset(c, 0, sizeof(*c));
    c->cache_dir = strdup(cache_dir);
    c->max_size = max_size;
}

void
cache_close(ModelCache *c)
{
    cache_clear(c);
    free(c->cache_dir);
    memset(c, 0, sizeof(*c));
}

/* Helper: create directory tree */
static int
mkdirp(char *path)
{
    char *p, *q;
    char tmp[512];
    int fd;

    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (access(tmp, AEXIST) < 0) {
                fd = create(tmp, OREAD, DMDIR | 0755);
                if (fd >= 0)
                    close(fd);
            }
            *p = '/';
        }
    }

    /* Create final directory */
    if (access(tmp, AEXIST) < 0) {
        fd = create(tmp, OREAD, DMDIR | 0755);
        if (fd >= 0) {
            close(fd);
            return 0;
        }
        return -1;
    }
    return 0;
}

/* Helper: get file size */
static vlong
filesize(char *path)
{
    Dir *d;
    vlong size;

    d = dirstat(path);
    if (d == nil)
        return -1;
    size = d->length;
    free(d);
    return size;
}

int
cache_scan(ModelCache *c)
{
    Dir *d, *entries;
    int fd, n, i;
    char path[512];
    char subpath[512];
    vlong size;
    CacheEntry *e;

    /* Open cache directory */
    fd = open(c->cache_dir, OREAD);
    if (fd < 0)
        return -1;

    /* Read directory entries (repos) */
    n = dirreadall(fd, &entries);
    close(fd);

    if (n < 0)
        return -1;

    /* Scan each repo directory */
    for (i = 0; i < n; i++) {
        d = &entries[i];
        if (!(d->mode & DMDIR))
            continue;

        snprint(path, sizeof path, "%s/%s", c->cache_dir, d->name);

        /* Open repo directory */
        int rfd = open(path, OREAD);
        if (rfd < 0)
            continue;

        Dir *files;
        int nf = dirreadall(rfd, &files);
        close(rfd);

        if (nf < 0)
            continue;

        /* Add each file to cache index */
        for (int j = 0; j < nf; j++) {
            if (files[j].mode & DMDIR)
                continue;

            snprint(subpath, sizeof subpath, "%s/%s", path, files[j].name);
            size = files[j].length;

            e = malloc(sizeof(*e));
            if (e == nil)
                continue;

            memset(e, 0, sizeof(*e));
            e->repo_id = strdup(d->name);
            e->filename = strdup(files[j].name);
            e->localpath = strdup(subpath);
            e->size = size;
            e->atime = files[j].atime;

            /* Add to front of LRU list */
            e->next = c->head;
            e->prev = nil;
            if (c->head)
                c->head->prev = e;
            c->head = e;
            if (c->tail == nil)
                c->tail = e;

            c->total_size += size;
            c->num_entries++;
        }
        free(files);
    }
    free(entries);

    return 0;
}

char *
cache_lookup(ModelCache *c, char *repo_id, char *filename)
{
    CacheEntry *e;

    e = cache_find(c, repo_id, filename);
    if (e == nil)
        return nil;

    /* Touch to update LRU order */
    cache_touch(c, e);

    return strdup(e->localpath);
}

int
cache_add(ModelCache *c, char *repo_id, char *filename,
          char *localpath, vlong size)
{
    CacheEntry *e;

    /* Check if already in cache */
    e = cache_find(c, repo_id, filename);
    if (e != nil) {
        cache_touch(c, e);
        return 0;
    }

    /* Evict if needed */
    if (c->max_size > 0 && c->total_size + size > c->max_size) {
        if (cache_evict(c, size) < 0)
            return -1;
    }

    /* Create new entry */
    e = malloc(sizeof(*e));
    if (e == nil)
        return -1;

    memset(e, 0, sizeof(*e));
    e->repo_id = strdup(repo_id);
    e->filename = strdup(filename);
    e->localpath = strdup(localpath);
    e->size = size;
    e->atime = nsec();

    /* Add to front of LRU list */
    e->next = c->head;
    e->prev = nil;
    if (c->head)
        c->head->prev = e;
    c->head = e;
    if (c->tail == nil)
        c->tail = e;

    c->total_size += size;
    c->num_entries++;

    return 0;
}

int
cache_remove(ModelCache *c, char *repo_id, char *filename)
{
    CacheEntry *e;

    e = cache_find(c, repo_id, filename);
    if (e == nil)
        return -1;

    /* Remove from LRU list */
    if (e->prev)
        e->prev->next = e->next;
    else
        c->head = e->next;

    if (e->next)
        e->next->prev = e->prev;
    else
        c->tail = e->prev;

    c->total_size -= e->size;
    c->num_entries--;

    /* Delete file */
    remove(e->localpath);

    /* Free entry */
    free(e->repo_id);
    free(e->filename);
    free(e->localpath);
    free(e);

    return 0;
}

int
cache_evict(ModelCache *c, vlong new_size)
{
    CacheEntry *e, *prev;
    vlong target = c->max_size - new_size;

    if (target < 0)
        target = 0;

    /* Evict from tail (LRU) until we have space */
    e = c->tail;
    while (e != nil && c->total_size > target) {
        prev = e->prev;

        /* Remove from list */
        if (e->prev)
            e->prev->next = e->next;
        else
            c->head = e->next;

        if (e->next)
            e->next->prev = e->prev;
        else
            c->tail = e->prev;

        c->total_size -= e->size;
        c->num_entries--;

        /* Delete file */
        remove(e->localpath);

        /* Free entry */
        free(e->repo_id);
        free(e->filename);
        free(e->localpath);
        free(e);

        e = prev;
    }

    return 0;
}

void
cache_stats(ModelCache *c, vlong *total_size, int *num_entries)
{
    if (total_size)
        *total_size = c->total_size;
    if (num_entries)
        *num_entries = c->num_entries;
}

void
cache_touch(ModelCache *c, CacheEntry *e)
{
    if (e == c->head)
        return;  /* Already at front */

    /* Remove from current position */
    if (e->prev)
        e->prev->next = e->next;
    if (e->next)
        e->next->prev = e->prev;
    else
        c->tail = e->prev;

    /* Move to front */
    e->next = c->head;
    e->prev = nil;
    if (c->head)
        c->head->prev = e;
    c->head = e;
    if (c->tail == nil)
        c->tail = e;

    e->atime = nsec();
}

CacheEntry *
cache_find(ModelCache *c, char *repo_id, char *filename)
{
    CacheEntry *e;

    for (e = c->head; e != nil; e = e->next) {
        if (strcmp(e->repo_id, repo_id) == 0 &&
            strcmp(e->filename, filename) == 0)
            return e;
    }
    return nil;
}

CacheEntry *
cache_list_repo(ModelCache *c, char *repo_id)
{
    CacheEntry *e, *match, *head = nil, *tail = nil;

    for (e = c->head; e != nil; e = e->next) {
        if (strcmp(e->repo_id, repo_id) == 0) {
            match = malloc(sizeof(*match));
            if (match) {
                *match = *e;
                match->repo_id = strdup(e->repo_id);
                match->filename = strdup(e->filename);
                match->localpath = strdup(e->localpath);
                match->next = nil;
                match->prev = nil;

                if (tail) {
                    tail->next = match;
                    match->prev = tail;
                    tail = match;
                } else {
                    head = tail = match;
                }
            }
        }
    }

    return head;
}

void
cache_clear(ModelCache *c)
{
    CacheEntry *e, *next;

    for (e = c->head; e != nil; e = next) {
        next = e->next;
        free(e->repo_id);
        free(e->filename);
        free(e->localpath);
        free(e);
    }

    c->head = nil;
    c->tail = nil;
    c->total_size = 0;
    c->num_entries = 0;
}
