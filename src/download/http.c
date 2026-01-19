/*
 * http.c - Simple HTTP client for Plan 9
 *
 * Uses dial() for TCP connections and webfs for HTTPS.
 * webfs handles TLS transparently, avoiding fork/rfork issues in libthread.
 */

#include <u.h>
#include <libc.h>

#include "http.h"

/* Buffer sizes */
#define HTTP_BUFSIZE 8192
#define HTTP_LINEBUF 4096

/* WebFS paths */
#define WEBFS_CLONE "/mnt/web/clone"
#define WEBFS_DIR "/mnt/web"

/* Check if webfs is mounted */
static int
webfs_available(void)
{
    Dir *d = dirstat(WEBFS_CLONE);
    if (d == nil)
        return 0;
    free(d);
    return 1;
}

void
http_init(HttpClient *c)
{
    memset(c, 0, sizeof(*c));
    c->fd = -1;
    c->wfd = -1;
}

/*
 * WebFS-based HTTPS connection.
 * webfs handles TLS transparently.
 */
static int
http_connect_webfs(HttpClient *c, char *host, int port, int tls)
{
    USED(port);
    USED(tls);

    /* Just store host for later - webfs connections are per-request */
    c->host = strdup(host);
    c->port = port;
    c->tls = tls;
    c->fd = -2;  /* Special marker for webfs mode */
    c->wfd = -1;

    return 0;
}

/*
 * Direct TCP connection (for plain HTTP only).
 */
static int
http_connect_direct(HttpClient *c, char *host, int port)
{
    char addr[256];
    int fd;

    snprint(addr, sizeof addr, "tcp!%s!%d", host, port);

    fd = dial(addr, nil, nil, nil);
    if (fd < 0) {
        c->errmsg = "dial failed";
        return -1;
    }

    c->fd = fd;
    c->wfd = -1;  /* Same fd for read/write */
    c->tls = 0;
    c->host = strdup(host);
    c->port = port;

    return 0;
}

int
http_connect(HttpClient *c, char *host, int port, int tls)
{
    if (tls) {
        /* Use webfs for HTTPS - it handles TLS transparently */
        if (!webfs_available()) {
            /* Try to start webfs */
            int pid = fork();
            if (pid == 0) {
                execl("/bin/webfs", "webfs", nil);
                exits("exec webfs");
            }
            if (pid > 0) {
                sleep(1000);  /* Wait for webfs to start */
            }
            if (!webfs_available()) {
                c->errmsg = "webfs not available";
                return -1;
            }
        }
        return http_connect_webfs(c, host, port, tls);
    } else {
        return http_connect_direct(c, host, port);
    }
}

void
http_close(HttpClient *c)
{
    if (c->fd >= 0) {
        close(c->fd);
        c->fd = -1;
    }
    if (c->wfd >= 0) {
        close(c->wfd);
        c->wfd = -1;
    }
    free(c->host);
    c->host = nil;
}

/* Read a line from the connection (CRLF terminated) */
static int
http_readline(int fd, char *buf, int bufsize)
{
    int i = 0;
    char ch;

    while (i < bufsize - 1) {
        if (read(fd, &ch, 1) != 1)
            break;
        if (ch == '\r')
            continue;
        if (ch == '\n')
            break;
        buf[i++] = ch;
    }
    buf[i] = '\0';
    return i;
}

/* Parse HTTP status line */
static int
http_parse_status(char *line)
{
    char *p;

    /* Skip HTTP/1.x */
    p = strchr(line, ' ');
    if (p == nil)
        return -1;
    return atoi(p + 1);
}

/* Parse a single header line */
static void
http_parse_header(char *line, HttpResponse *resp)
{
    char *colon, *value;
    int len;

    colon = strchr(line, ':');
    if (colon == nil)
        return;

    *colon = '\0';
    value = colon + 1;
    while (*value == ' ' || *value == '\t')
        value++;

    /* Remove trailing whitespace */
    len = strlen(value);
    while (len > 0 && (value[len-1] == ' ' || value[len-1] == '\t'))
        value[--len] = '\0';

    if (cistrcmp(line, "Content-Length") == 0) {
        resp->content_length = strtoll(value, nil, 10);
    } else if (cistrcmp(line, "Content-Type") == 0) {
        resp->content_type = strdup(value);
    } else if (cistrcmp(line, "Location") == 0) {
        resp->location = strdup(value);
    } else if (cistrcmp(line, "Accept-Ranges") == 0) {
        if (cistrcmp(value, "bytes") == 0)
            resp->accept_ranges = 1;
    }
}

/*
 * HTTP GET via hget command.
 * hget handles HTTPS/TLS transparently.
 * We run hget to download to a temp file, then open it for reading.
 */
static int
http_get_hget(HttpClient *c, char *path, HttpHeader *headers, HttpResponse *resp)
{
    char url[1024];
    char tmpfile[64];
    int pid, fd;
    Waitmsg *w;
    static int tmpcount = 0;

    USED(headers);  /* hget doesn't support custom headers easily */

    memset(resp, 0, sizeof(*resp));
    resp->content_length = -1;

    /* Build URL */
    snprint(url, sizeof url, "https://%s%s", c->host, path);

    /* Create unique temp file */
    snprint(tmpfile, sizeof tmpfile, "/tmp/hget_%d_%d", getpid(), tmpcount++);

    /* Run hget */
    pid = fork();
    if (pid == 0) {
        /* Child: run hget, redirect stderr to /dev/null */
        int null = open("/dev/null", OWRITE);
        if (null >= 0) {
            dup(null, 2);
            close(null);
        }
        execl("/bin/hget", "hget", "-o", tmpfile, url, nil);
        exits("exec hget failed");
    }

    if (pid < 0) {
        c->errmsg = "fork failed";
        return -1;
    }

    /* Wait for hget to complete (with timeout) */
    for (int i = 0; i < 60; i++) {
        w = wait();
        if (w != nil) {
            if (w->pid == pid) {
                if (w->msg[0] != '\0') {
                    static char errbuf[128];
                    snprint(errbuf, sizeof errbuf, "hget: %s", w->msg);
                    c->errmsg = errbuf;
                    free(w);
                    return -1;
                }
                free(w);
                break;
            }
            free(w);
        }
        sleep(1000);
    }

    /* Open temp file for reading */
    fd = open(tmpfile, OREAD);
    if (fd < 0) {
        c->errmsg = "cannot open hget output";
        return -1;
    }

    /* Get file size for content_length */
    Dir *d = dirstat(tmpfile);
    if (d != nil) {
        resp->content_length = d->length;
        free(d);
    }

    resp->status = HTTP_OK;
    c->fd = fd;
    c->wfd = -1;

    /* Store temp file path for cleanup */
    c->host = strdup(tmpfile);  /* Reuse host field for temp file path */

    return 0;
}

/*
 * HTTP GET via direct socket.
 */
static int
http_get_direct(HttpClient *c, char *path, HttpHeader *headers, HttpResponse *resp)
{
    char buf[HTTP_BUFSIZE];
    char line[HTTP_LINEBUF];
    int n;
    HttpHeader *h;

    memset(resp, 0, sizeof(*resp));
    resp->content_length = -1;

    /* Build request */
    n = snprint(buf, sizeof buf,
                "GET %s HTTP/1.1\r\n"
                "Host: %s\r\n"
                "Connection: close\r\n"
                "User-Agent: 9ml/1.0\r\n",
                path, c->host);

    /* Add custom headers */
    for (h = headers; h != nil; h = h->next) {
        n += snprint(buf + n, sizeof(buf) - n,
                     "%s: %s\r\n", h->name, h->value);
    }

    n += snprint(buf + n, sizeof(buf) - n, "\r\n");

    /* Send request */
    if (write(c->fd, buf, n) != n) {
        c->errmsg = "write failed";
        return -1;
    }

    /* Read status line */
    if (http_readline(c->fd, line, sizeof line) <= 0) {
        c->errmsg = "no response";
        return -1;
    }
    resp->status = http_parse_status(line);
    if (resp->status < 0) {
        c->errmsg = "invalid status line";
        return -1;
    }

    /* Read headers */
    while (http_readline(c->fd, line, sizeof line) > 0) {
        http_parse_header(line, resp);
    }

    return 0;
}

int
http_get(HttpClient *c, char *path, HttpHeader *headers, HttpResponse *resp)
{
    if (c->fd == -2) {
        /* TLS mode - use hget */
        return http_get_hget(c, path, headers, resp);
    } else {
        /* Direct socket mode (plain HTTP) */
        return http_get_direct(c, path, headers, resp);
    }
}

int
http_get_range(HttpClient *c, char *path, vlong start, vlong end,
               HttpHeader *headers, HttpResponse *resp)
{
    HttpHeader *range_hdr, *h;
    char range_val[64];
    int ret;

    /* Build Range header */
    if (end >= 0)
        snprint(range_val, sizeof range_val, "bytes=%lld-%lld", start, end);
    else
        snprint(range_val, sizeof range_val, "bytes=%lld-", start);

    range_hdr = http_header_add(nil, "Range", range_val);

    /* Append existing headers */
    if (headers != nil) {
        for (h = range_hdr; h->next != nil; h = h->next)
            ;
        h->next = headers;
    }

    ret = http_get(c, path, range_hdr, resp);

    /* Free just the range header (not the user's headers) */
    range_hdr->next = nil;
    http_headers_free(range_hdr);

    return ret;
}

int
http_download(HttpClient *c, char *path, char *localpath,
              HttpProgressFn progress, void *arg)
{
    return http_download_resume(c, path, localpath, 0, progress, arg);
}

int
http_download_resume(HttpClient *c, char *path, char *localpath,
                     vlong offset, HttpProgressFn progress, void *arg)
{
    HttpResponse resp;
    char buf[HTTP_BUFSIZE];
    int fd, n;
    vlong downloaded, total;
    int flags;

    /* Open local file */
    flags = OWRITE;
    if (offset > 0)
        flags |= OTRUNC;  /* Append mode not directly supported, use seek */
    else
        flags = OWRITE | OTRUNC;

    fd = create(localpath, flags, 0644);
    if (fd < 0) {
        /* Try opening existing file */
        fd = open(localpath, OWRITE);
        if (fd < 0) {
            c->errmsg = "cannot open local file";
            return -1;
        }
    }

    /* Seek to offset if resuming */
    if (offset > 0) {
        if (seek(fd, offset, 0) != offset) {
            close(fd);
            c->errmsg = "seek failed";
            return -1;
        }
    }

    /* Send request */
    if (offset > 0) {
        if (http_get_range(c, path, offset, -1, nil, &resp) < 0) {
            close(fd);
            return -1;
        }
        if (resp.status != HTTP_PARTIAL && resp.status != HTTP_OK) {
            close(fd);
            http_resp_free(&resp);
            c->errmsg = "server does not support resume";
            return -1;
        }
    } else {
        if (http_get(c, path, nil, &resp) < 0) {
            close(fd);
            return -1;
        }
        if (resp.status != HTTP_OK) {
            close(fd);
            http_resp_free(&resp);
            c->errmsg = "download failed";
            return -1;
        }
    }

    total = resp.content_length;
    if (total > 0 && offset > 0)
        total += offset;

    downloaded = offset;

    /* Read body and write to file */
    while ((n = read(c->fd, buf, sizeof buf)) > 0) {
        if (write(fd, buf, n) != n) {
            close(fd);
            http_resp_free(&resp);
            c->errmsg = "write to file failed";
            return -1;
        }
        downloaded += n;
        if (progress)
            progress(downloaded, total, arg);
    }

    close(fd);
    http_resp_free(&resp);
    return 0;
}

void
http_resp_free(HttpResponse *resp)
{
    free(resp->content_type);
    free(resp->location);
    free(resp->body);
    memset(resp, 0, sizeof(*resp));
}

void
http_headers_free(HttpHeader *h)
{
    HttpHeader *next;

    while (h != nil) {
        next = h->next;
        free(h->name);
        free(h->value);
        free(h);
        h = next;
    }
}

HttpHeader *
http_header_add(HttpHeader *list, char *name, char *value)
{
    HttpHeader *h;

    h = malloc(sizeof(*h));
    if (h == nil)
        return list;

    h->name = strdup(name);
    h->value = strdup(value);
    h->next = list;

    return h;
}

/* URL parsing */
int
http_parse_url(char *url, HttpUrl *u)
{
    char *p, *q, *path_start;
    int len;

    memset(u, 0, sizeof(*u));

    /* Parse scheme */
    p = strstr(url, "://");
    if (p == nil) {
        u->scheme = strdup("http");
        p = url;
    } else {
        len = p - url;
        u->scheme = malloc(len + 1);
        memmove(u->scheme, url, len);
        u->scheme[len] = '\0';
        p += 3;
    }

    /* Default port based on scheme */
    if (strcmp(u->scheme, "https") == 0)
        u->port = 443;
    else
        u->port = 80;

    /* Find end of host (start of path, query, or end of string) */
    path_start = strchr(p, '/');
    q = strchr(p, '?');

    if (path_start == nil)
        path_start = p + strlen(p);
    if (q != nil && q < path_start)
        path_start = q;

    /* Check for port */
    q = strchr(p, ':');
    if (q != nil && q < path_start) {
        /* Host with port */
        len = q - p;
        u->host = malloc(len + 1);
        memmove(u->host, p, len);
        u->host[len] = '\0';
        u->port = atoi(q + 1);
    } else {
        /* Host without port */
        len = path_start - p;
        u->host = malloc(len + 1);
        memmove(u->host, p, len);
        u->host[len] = '\0';
    }

    /* Parse path */
    if (*path_start == '/') {
        q = strchr(path_start, '?');
        if (q != nil) {
            len = q - path_start;
            u->path = malloc(len + 1);
            memmove(u->path, path_start, len);
            u->path[len] = '\0';
            u->query = strdup(q + 1);
        } else {
            u->path = strdup(path_start);
        }
    } else if (*path_start == '?') {
        u->path = strdup("/");
        u->query = strdup(path_start + 1);
    } else {
        u->path = strdup("/");
    }

    return 0;
}

void
http_url_free(HttpUrl *u)
{
    free(u->scheme);
    free(u->host);
    free(u->path);
    free(u->query);
    memset(u, 0, sizeof(*u));
}
