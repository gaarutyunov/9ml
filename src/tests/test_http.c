/*
 * test_http.c - Test HTTP URL parsing and header handling
 *
 * Tests URL parsing, header management, and JSON parsing helpers.
 * Note: Actual network tests would require internet access in VM.
 */

/* Include implementation directly for testing.
 * Note: http.c includes <u.h> and <libc.h>, so we don't include them again. */
#include "download/http.c"

void
main(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;
    HttpUrl url;
    HttpHeader *headers;

    USED(argc);
    USED(argv);

    print("=== HTTP URL Parsing Test ===\n");

    /* Test 1: Parse simple HTTPS URL */
    print("\nTest 1: Simple HTTPS URL\n");
    if (http_parse_url("https://huggingface.co/api/models", &url) == 0) {
        int ok = 1;
        if (strcmp(url.scheme, "https") != 0) {
            print("  scheme mismatch: %s\n", url.scheme);
            ok = 0;
        }
        if (strcmp(url.host, "huggingface.co") != 0) {
            print("  host mismatch: %s\n", url.host);
            ok = 0;
        }
        if (url.port != 443) {
            print("  port mismatch: %d\n", url.port);
            ok = 0;
        }
        if (strcmp(url.path, "/api/models") != 0) {
            print("  path mismatch: %s\n", url.path);
            ok = 0;
        }
        http_url_free(&url);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    } else {
        print("  Result: FAIL (parse failed)\n");
        failed++;
    }

    /* Test 2: Parse HTTP URL with port */
    print("\nTest 2: HTTP URL with port\n");
    if (http_parse_url("http://localhost:8080/path/to/file", &url) == 0) {
        int ok = 1;
        if (strcmp(url.scheme, "http") != 0) {
            print("  scheme mismatch: %s\n", url.scheme);
            ok = 0;
        }
        if (strcmp(url.host, "localhost") != 0) {
            print("  host mismatch: %s\n", url.host);
            ok = 0;
        }
        if (url.port != 8080) {
            print("  port mismatch: %d\n", url.port);
            ok = 0;
        }
        if (strcmp(url.path, "/path/to/file") != 0) {
            print("  path mismatch: %s\n", url.path);
            ok = 0;
        }
        http_url_free(&url);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    } else {
        print("  Result: FAIL (parse failed)\n");
        failed++;
    }

    /* Test 3: Parse URL with query string */
    print("\nTest 3: URL with query string\n");
    if (http_parse_url("https://example.com/search?q=test&page=1", &url) == 0) {
        int ok = 1;
        if (strcmp(url.host, "example.com") != 0) {
            print("  host mismatch: %s\n", url.host);
            ok = 0;
        }
        if (strcmp(url.path, "/search") != 0) {
            print("  path mismatch: %s\n", url.path);
            ok = 0;
        }
        if (url.query == nil || strcmp(url.query, "q=test&page=1") != 0) {
            print("  query mismatch: %s\n", url.query);
            ok = 0;
        }
        http_url_free(&url);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    } else {
        print("  Result: FAIL (parse failed)\n");
        failed++;
    }

    /* Test 4: Parse URL without scheme */
    print("\nTest 4: URL without scheme\n");
    if (http_parse_url("example.com/path", &url) == 0) {
        int ok = 1;
        if (strcmp(url.scheme, "http") != 0) {
            print("  scheme mismatch: %s\n", url.scheme);
            ok = 0;
        }
        if (strcmp(url.host, "example.com") != 0) {
            print("  host mismatch: %s\n", url.host);
            ok = 0;
        }
        http_url_free(&url);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    } else {
        print("  Result: FAIL (parse failed)\n");
        failed++;
    }

    /* Test 5: Header list management */
    print("\nTest 5: Header list management\n");
    {
        int ok = 1;
        headers = nil;
        headers = http_header_add(headers, "Content-Type", "application/json");
        headers = http_header_add(headers, "Authorization", "Bearer token123");

        if (headers == nil) {
            print("  headers is nil\n");
            ok = 0;
        } else {
            /* Check first header (should be Authorization since we prepend) */
            if (strcmp(headers->name, "Authorization") != 0) {
                print("  first header name mismatch: %s\n", headers->name);
                ok = 0;
            }
            if (strcmp(headers->value, "Bearer token123") != 0) {
                print("  first header value mismatch: %s\n", headers->value);
                ok = 0;
            }
            /* Check second header */
            if (headers->next == nil) {
                print("  second header missing\n");
                ok = 0;
            } else if (strcmp(headers->next->name, "Content-Type") != 0) {
                print("  second header name mismatch: %s\n", headers->next->name);
                ok = 0;
            }
        }

        http_headers_free(headers);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    }

    /* Test 6: URL with only host */
    print("\nTest 6: URL with only host\n");
    if (http_parse_url("https://api.github.com", &url) == 0) {
        int ok = 1;
        if (strcmp(url.host, "api.github.com") != 0) {
            print("  host mismatch: %s\n", url.host);
            ok = 0;
        }
        if (strcmp(url.path, "/") != 0) {
            print("  path mismatch: %s (expected /)\n", url.path);
            ok = 0;
        }
        http_url_free(&url);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    } else {
        print("  Result: FAIL (parse failed)\n");
        failed++;
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d HTTP tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
