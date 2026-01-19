/*
 * hfhub.c - HuggingFace Hub API client
 *
 * Downloads models from HuggingFace Hub using the REST API.
 */

#include <u.h>
#include <libc.h>

#include "http.h"
#include "hfhub.h"

/* JSON parsing helpers (minimal implementation) */

/* Skip whitespace */
static char *
json_skip_ws(char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    return p;
}

/* Parse a JSON string value, returns pointer after closing quote */
static char *
json_parse_string(char *p, char **out)
{
    char *start, *end;
    int len;

    p = json_skip_ws(p);
    if (*p != '"')
        return nil;

    start = ++p;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1])
            p++;
        p++;
    }
    if (*p != '"')
        return nil;

    end = p++;
    len = end - start;
    *out = malloc(len + 1);
    if (*out) {
        memmove(*out, start, len);
        (*out)[len] = '\0';
    }
    return p;
}

/* Parse a JSON number value */
static char *
json_parse_number(char *p, vlong *out)
{
    p = json_skip_ws(p);
    *out = strtoll(p, &p, 10);
    return p;
}

/* Skip a JSON value (string, number, object, array, true, false, null) */
static char *
json_skip_value(char *p)
{
    int depth;

    p = json_skip_ws(p);

    switch (*p) {
    case '"':
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1])
                p++;
            p++;
        }
        if (*p == '"')
            p++;
        break;

    case '[':
    case '{':
        depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[' || *p == '{')
                depth++;
            else if (*p == ']' || *p == '}')
                depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1])
                        p++;
                    p++;
                }
            }
            if (*p)
                p++;
        }
        break;

    default:
        /* number, true, false, null */
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            p++;
        break;
    }

    return p;
}

/* Find a key in a JSON object, returns pointer to value or nil */
static char *
json_find_key(char *p, char *key)
{
    char *found_key;
    int keylen = strlen(key);

    p = json_skip_ws(p);
    if (*p != '{')
        return nil;
    p++;

    while (*p && *p != '}') {
        p = json_skip_ws(p);
        if (*p != '"')
            break;

        /* Parse key */
        p = json_parse_string(p, &found_key);
        if (p == nil)
            break;

        p = json_skip_ws(p);
        if (*p != ':') {
            free(found_key);
            break;
        }
        p++;
        p = json_skip_ws(p);

        /* Check if this is the key we want */
        if (strlen(found_key) == keylen && strcmp(found_key, key) == 0) {
            free(found_key);
            return p;
        }
        free(found_key);

        /* Skip value */
        p = json_skip_value(p);
        p = json_skip_ws(p);
        if (*p == ',')
            p++;
    }

    return nil;
}

void
hf_init(HFHub *h, char *cache_dir)
{
    memset(h, 0, sizeof(*h));
    http_init(&h->http);
    if (cache_dir)
        h->cache_dir = strdup(cache_dir);
    else
        h->cache_dir = strdup("/tmp/hf_cache");
}

void
hf_set_token(HFHub *h, char *token)
{
    free(h->token);
    h->token = token ? strdup(token) : nil;
}

void
hf_close(HFHub *h)
{
    http_close(&h->http);
    free(h->token);
    free(h->cache_dir);
    memset(h, 0, sizeof(*h));
}

int
hf_parse_file_list(char *json, HFFile **files)
{
    char *p, *item_start;
    HFFile *file, *head = nil, *tail = nil;
    char *val;
    vlong num;

    *files = nil;

    p = json_skip_ws(json);
    if (*p != '[')
        return -1;
    p++;

    while (*p && *p != ']') {
        p = json_skip_ws(p);
        if (*p != '{')
            break;

        item_start = p;

        file = malloc(sizeof(*file));
        if (file == nil)
            break;
        memset(file, 0, sizeof(*file));

        /* Parse filename/path */
        val = json_find_key(item_start, "path");
        if (val) {
            json_parse_string(val, &file->path);
            /* Extract filename from path */
            char *slash = strrchr(file->path, '/');
            file->filename = strdup(slash ? slash + 1 : file->path);
        }

        /* Parse size */
        val = json_find_key(item_start, "size");
        if (val) {
            json_parse_number(val, &num);
            file->size = num;
        }

        /* Check for LFS info */
        val = json_find_key(item_start, "lfs");
        if (val && *val == '{') {
            file->is_lfs = 1;
            char *lfs_val = json_find_key(val, "sha256");
            if (lfs_val)
                json_parse_string(lfs_val, &file->sha256);
        }

        /* Add to list */
        if (file->path) {
            if (tail) {
                tail->next = file;
                tail = file;
            } else {
                head = tail = file;
            }
        } else {
            free(file);
        }

        /* Skip to next item */
        p = json_skip_value(item_start);
        p = json_skip_ws(p);
        if (*p == ',')
            p++;
    }

    *files = head;
    return 0;
}

int
hf_get_model_info(HFHub *h, char *repo_id, char *revision, HFModel *model)
{
    char path[512];
    char buf[65536];  /* Buffer for API response */
    HttpResponse resp;
    HttpHeader *headers = nil;
    int n, total;

    memset(model, 0, sizeof(*model));
    model->repo_id = strdup(repo_id);
    model->revision = strdup(revision ? revision : "main");

    /* Connect to HuggingFace API */
    if (http_connect(&h->http, HF_API_HOST, 443, 1) < 0) {
        h->errmsg = h->http.errmsg;
        return -1;
    }

    /* Add auth header if token is set */
    if (h->token)
        headers = http_header_add(headers, "Authorization",
                                  smprint("Bearer %s", h->token));

    /* Get file listing */
    snprint(path, sizeof path, "/api/models/%s/tree/%s",
            repo_id, model->revision);

    if (http_get(&h->http, path, headers, &resp) < 0) {
        h->errmsg = h->http.errmsg;
        http_headers_free(headers);
        http_close(&h->http);
        return -1;
    }

    if (resp.status != HTTP_OK) {
        h->errmsg = "API request failed";
        http_resp_free(&resp);
        http_headers_free(headers);
        http_close(&h->http);
        return -1;
    }

    /* Read response body */
    total = 0;
    while ((n = read(h->http.fd, buf + total, sizeof(buf) - total - 1)) > 0) {
        total += n;
        if (total >= sizeof(buf) - 1)
            break;
    }
    buf[total] = '\0';

    http_resp_free(&resp);
    http_headers_free(headers);
    http_close(&h->http);

    /* Parse file list */
    if (hf_parse_file_list(buf, &model->files) < 0) {
        h->errmsg = "failed to parse API response";
        return -1;
    }

    return 0;
}

HFFile *
hf_find_files(HFModel *model, char *pattern)
{
    HFFile *f, *match, *head = nil, *tail = nil;
    int plen;

    if (pattern == nil || *pattern == '\0')
        return nil;

    plen = strlen(pattern);

    for (f = model->files; f != nil; f = f->next) {
        /* Simple suffix match (e.g., ".gguf") */
        if (pattern[0] == '.') {
            int flen = strlen(f->filename);
            if (flen >= plen &&
                strcmp(f->filename + flen - plen, pattern) == 0) {
                match = malloc(sizeof(*match));
                if (match) {
                    *match = *f;
                    match->filename = strdup(f->filename);
                    match->path = strdup(f->path);
                    if (f->sha256)
                        match->sha256 = strdup(f->sha256);
                    if (f->lfs_url)
                        match->lfs_url = strdup(f->lfs_url);
                    match->next = nil;

                    if (tail) {
                        tail->next = match;
                        tail = match;
                    } else {
                        head = tail = match;
                    }
                }
            }
        }
        /* Substring match */
        else if (strstr(f->filename, pattern) != nil) {
            match = malloc(sizeof(*match));
            if (match) {
                *match = *f;
                match->filename = strdup(f->filename);
                match->path = strdup(f->path);
                if (f->sha256)
                    match->sha256 = strdup(f->sha256);
                if (f->lfs_url)
                    match->lfs_url = strdup(f->lfs_url);
                match->next = nil;

                if (tail) {
                    tail->next = match;
                    tail = match;
                } else {
                    head = tail = match;
                }
            }
        }
    }

    return head;
}

int
hf_download_file(HFHub *h, HFModel *model, HFFile *file,
                 HttpProgressFn progress, void *arg)
{
    char localpath[512];

    /* Build cache path */
    snprint(localpath, sizeof localpath, "%s/%s/%s",
            h->cache_dir, model->repo_id, file->filename);

    return hf_download_file_to(h, model, file, localpath, progress, arg);
}

int
hf_download_file_to(HFHub *h, HFModel *model, HFFile *file,
                    char *localpath, HttpProgressFn progress, void *arg)
{
    char path[512];
    char *dir;
    HttpHeader *headers = nil;
    int ret;

    /* Create directory if needed */
    dir = strdup(localpath);
    if (dir) {
        char *slash = strrchr(dir, '/');
        if (slash) {
            *slash = '\0';
            /* Create directory tree - simplified, assumes parent exists */
            if (access(dir, AEXIST) < 0) {
                int dfd = create(dir, OREAD, DMDIR | 0755);
                if (dfd >= 0)
                    close(dfd);
            }
        }
        free(dir);
    }

    /* Connect to HuggingFace */
    if (http_connect(&h->http, HF_API_HOST, 443, 1) < 0) {
        h->errmsg = h->http.errmsg;
        return -1;
    }

    /* Add auth header if token is set */
    if (h->token)
        headers = http_header_add(headers, "Authorization",
                                  smprint("Bearer %s", h->token));

    /* Build download path */
    snprint(path, sizeof path, "/%s/resolve/%s/%s",
            model->repo_id, model->revision, file->path);

    ret = http_download(&h->http, path, localpath, progress, arg);
    if (ret < 0)
        h->errmsg = h->http.errmsg;

    http_headers_free(headers);
    http_close(&h->http);

    return ret;
}

char *
hf_get_cached_path(HFHub *h, char *repo_id, char *filename)
{
    char path[512];

    snprint(path, sizeof path, "%s/%s/%s", h->cache_dir, repo_id, filename);
    if (access(path, AREAD) == 0)
        return strdup(path);
    return nil;
}

void
hf_model_free(HFModel *model)
{
    free(model->repo_id);
    free(model->revision);
    free(model->pipeline_tag);
    free(model->model_type);
    hf_files_free(model->files);
    memset(model, 0, sizeof(*model));
}

void
hf_files_free(HFFile *files)
{
    HFFile *next;

    while (files) {
        next = files->next;
        free(files->filename);
        free(files->path);
        free(files->sha256);
        free(files->lfs_url);
        free(files);
        files = next;
    }
}

char *
hf_url_encode(char *s)
{
    static char *hex = "0123456789ABCDEF";
    char *out, *p;
    int len = 0;

    /* Calculate output length */
    for (p = s; *p; p++) {
        if ((*p >= 'A' && *p <= 'Z') ||
            (*p >= 'a' && *p <= 'z') ||
            (*p >= '0' && *p <= '9') ||
            *p == '-' || *p == '_' || *p == '.' || *p == '~')
            len++;
        else
            len += 3;
    }

    out = malloc(len + 1);
    if (out == nil)
        return nil;

    p = out;
    for (; *s; s++) {
        if ((*s >= 'A' && *s <= 'Z') ||
            (*s >= 'a' && *s <= 'z') ||
            (*s >= '0' && *s <= '9') ||
            *s == '-' || *s == '_' || *s == '.' || *s == '~') {
            *p++ = *s;
        } else {
            *p++ = '%';
            *p++ = hex[(*s >> 4) & 0xf];
            *p++ = hex[*s & 0xf];
        }
    }
    *p = '\0';

    return out;
}
