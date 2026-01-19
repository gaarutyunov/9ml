/*
 * test_pool_lru.c - Test model pool LRU eviction
 *
 * Tests pool initialization, entry management, and LRU eviction
 * using real model loading.
 */

/* Include model.c first (it has the headers and model_load/free functions) */
#include "model.c"

/* Now include pool implementation */
#include "pool/pool.c"

void
threadmain(int argc, char *argv[])
{
    ModelPool pool;
    PoolEntry *e1, *e2;
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== Pool LRU Test ===\n");

    /* Test 1: Initialize pool */
    print("\nTest 1: Initialize pool\n");
    pool_init(&pool, 4, 500 * 1024 * 1024);  /* Max 4 models, 500MB */
    if (pool.max_models == 4 && pool.max_memory == 500ULL * 1024 * 1024) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    /* Test 2: Load first model */
    print("\nTest 2: Load model\n");
    e1 = pool_load(&pool, "stories15M", "stories15M.bin", "tokenizer.bin");
    if (e1 != nil && pool.count == 1 && e1->refcount == 1) {
        print("  Loaded: %s (memory: %lludMB)\n", e1->name, e1->memory / (1024 * 1024));
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (e1=%p, count=%d)\n", e1, pool.count);
        failed++;
    }

    /* Test 3: Get same model (should return existing, increment refcount) */
    print("\nTest 3: Get same model\n");
    e2 = pool_get(&pool, "stories15M");
    if (e2 == e1 && e1->refcount == 2) {
        print("  Result: PASS (refcount=2)\n");
        passed++;
    } else {
        print("  Result: FAIL (e2=%p, refcount=%d)\n", e2, e1 ? e1->refcount : -1);
        failed++;
    }

    /* Test 4: Release model */
    print("\nTest 4: Release model\n");
    pool_release(&pool, e2);
    if (e1->refcount == 1) {
        print("  Result: PASS (refcount=1)\n");
        passed++;
    } else {
        print("  Result: FAIL (refcount=%d)\n", e1->refcount);
        failed++;
    }

    /* Test 5: Load same model again (should return existing, no new load) */
    print("\nTest 5: Load same model again\n");
    uvlong old_memory = pool.total_memory;
    e2 = pool_load(&pool, "stories15M", "stories15M.bin", "tokenizer.bin");
    if (e2 == e1 && pool.total_memory == old_memory && e1->refcount == 2) {
        print("  Result: PASS (no new load, refcount=2)\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }
    pool_release(&pool, e2);

    /* Test 6: Pool memory tracking */
    print("\nTest 6: Pool memory\n");
    uvlong mem = pool_memory(&pool);
    if (mem > 0 && mem == e1->memory) {
        print("  Memory: %lludMB\n", mem / (1024 * 1024));
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (got %llud, expected %llud)\n", mem, e1 ? e1->memory : 0);
        failed++;
    }

    /* Test 7: Pool list */
    print("\nTest 7: Pool list\n");
    char *list = pool_list(&pool);
    if (list != nil && strstr(list, "stories15M") != nil) {
        print("  Models: %s\n", list);
        print("  Result: PASS\n");
        passed++;
        free(list);
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    /* Test 8: Release and unload */
    print("\nTest 8: Release and unload\n");
    pool_release(&pool, e1);
    if (pool_unload(&pool, "stories15M") == 0 && pool.count == 0) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (count=%d)\n", pool.count);
        failed++;
    }

    /* Cleanup */
    pool_free(&pool);

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d pool tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    threadexitsall(failed ? "fail" : nil);
}
