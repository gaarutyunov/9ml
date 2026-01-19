/*
 * http.h - Simple HTTP client for Plan 9
 *
 * Uses dial() for connections and webfs or direct sockets for HTTP.
 * TLS is handled via tlsclient or the /sys/lib/tls/ca thumbprint system.
 */

#ifndef HTTP_H
#define HTTP_H

/* HTTP response status codes */
#define HTTP_OK 200
#define HTTP_PARTIAL 206
#define HTTP_MOVED 301
#define HTTP_FOUND 302
#define HTTP_NOT_MODIFIED 304
#define HTTP_BAD_REQUEST 400
#define HTTP_UNAUTHORIZED 401
#define HTTP_FORBIDDEN 403
#define HTTP_NOT_FOUND 404
#define HTTP_SERVER_ERROR 500

/* HTTP client state */
typedef struct HttpClient HttpClient;
struct HttpClient {
    int fd;             /* Connection file descriptor (read) */
    int wfd;            /* Write file descriptor (for TLS pipes, -1 if same as fd) */
    int tls;            /* 1 if TLS connection */
    char *host;         /* Server hostname */
    int port;           /* Server port */
    char *errmsg;       /* Last error message */
};

/* HTTP response */
typedef struct HttpResponse HttpResponse;
struct HttpResponse {
    int status;         /* HTTP status code */
    vlong content_length;   /* Content-Length header (-1 if chunked/unknown) */
    char *content_type;     /* Content-Type header */
    char *location;         /* Location header (for redirects) */
    int accept_ranges;      /* 1 if server supports Range requests */
    char *body;             /* Response body (for small responses) */
    vlong body_len;         /* Body length */
};

/* HTTP request headers */
typedef struct HttpHeader HttpHeader;
struct HttpHeader {
    char *name;
    char *value;
    HttpHeader *next;
};

/* Initialize HTTP client */
void http_init(HttpClient *c);

/* Connect to server (host:port, tls=1 for HTTPS) */
int http_connect(HttpClient *c, char *host, int port, int tls);

/* Close connection */
void http_close(HttpClient *c);

/* Send HTTP GET request with optional headers */
int http_get(HttpClient *c, char *path, HttpHeader *headers, HttpResponse *resp);

/* Send HTTP GET request with Range header for partial download */
int http_get_range(HttpClient *c, char *path, vlong start, vlong end,
                   HttpHeader *headers, HttpResponse *resp);

/* Download file to local path with progress callback */
typedef void (*HttpProgressFn)(vlong downloaded, vlong total, void *arg);
int http_download(HttpClient *c, char *path, char *localpath,
                  HttpProgressFn progress, void *arg);

/* Resume download from given offset */
int http_download_resume(HttpClient *c, char *path, char *localpath,
                         vlong offset, HttpProgressFn progress, void *arg);

/* Free response resources */
void http_resp_free(HttpResponse *resp);

/* Free header list */
void http_headers_free(HttpHeader *h);

/* Add header to list */
HttpHeader *http_header_add(HttpHeader *list, char *name, char *value);

/* URL parsing */
typedef struct HttpUrl HttpUrl;
struct HttpUrl {
    char *scheme;       /* http or https */
    char *host;
    int port;
    char *path;
    char *query;
};

int http_parse_url(char *url, HttpUrl *u);
void http_url_free(HttpUrl *u);

#endif /* HTTP_H */
