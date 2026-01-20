/*
 * test_config_json.c - Test config.json parsing for safetensors loader
 *
 * Tests the JSON parsing functionality that extracts model configuration
 * from HuggingFace config.json files.
 */

#include <u.h>
#include <libc.h>

/* Architecture IDs */
#define ARCH_UNKNOWN  0
#define ARCH_LLAMA2   1

/*
 * Simple JSON value parser - extracts a string value for a given key.
 * Returns pointer to start of value (after quotes for strings), or nil if not found.
 */
static char *
json_find_string(char *json, char *key)
{
    char search[128];
    char *p;

    snprint(search, sizeof(search), "\"%s\"", key);
    p = strstr(json, search);
    if (p == nil)
        return nil;

    /* Find the colon */
    p = strchr(p + strlen(search), ':');
    if (p == nil)
        return nil;

    /* Skip whitespace */
    p++;
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n'))
        p++;

    /* If it's a string, skip the opening quote */
    if (*p == '"')
        p++;

    return p;
}

/*
 * Extract integer value from JSON
 */
static int
json_get_int(char *json, char *key, int def)
{
    char *p = json_find_string(json, key);
    if (p == nil)
        return def;
    return strtol(p, nil, 10);
}

/*
 * Extract float value from JSON
 */
static float
json_get_float(char *json, char *key, float def)
{
    char *p = json_find_string(json, key);
    if (p == nil)
        return def;
    return strtod(p, nil);
}

/*
 * Check if model_type matches expected value
 */
static int
json_model_type_is(char *json, char *expected)
{
    char *p = json_find_string(json, "model_type");
    if (p == nil)
        return 0;
    return strncmp(p, expected, strlen(expected)) == 0;
}

void
main(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== config.json Parsing Test ===\n");

    /* Test 1: Parse model_type */
    print("\nTest 1: Parse model_type\n");
    {
        char *json = "{\"model_type\": \"llama\", \"hidden_size\": 288}";
        if (json_model_type_is(json, "llama")) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    }

    /* Test 2: Parse integer value */
    print("\nTest 2: Parse hidden_size (integer)\n");
    {
        char *json = "{\"model_type\": \"llama\", \"hidden_size\": 288}";
        int val = json_get_int(json, "hidden_size", 0);
        if (val == 288) {
            print("  Result: PASS (hidden_size=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 288, got %d)\n", val);
            failed++;
        }
    }

    /* Test 3: Parse float value (rope_theta) */
    print("\nTest 3: Parse rope_theta (float)\n");
    {
        char *json = "{\"rope_theta\": 10000.0, \"hidden_size\": 288}";
        float val = json_get_float(json, "rope_theta", 0.0f);
        float diff = val - 10000.0f;
        if (diff < 0) diff = -diff;
        if (diff < 1.0f) {
            print("  Result: PASS (rope_theta=%f)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 10000.0, got %f)\n", val);
            failed++;
        }
    }

    /* Test 4: Parse large rope_theta (500000) */
    print("\nTest 4: Parse large rope_theta (500000)\n");
    {
        char *json = "{\"rope_theta\": 500000.0, \"model_type\": \"llama\"}";
        float val = json_get_float(json, "rope_theta", 0.0f);
        float diff = val - 500000.0f;
        if (diff < 0) diff = -diff;
        if (diff < 1.0f) {
            print("  Result: PASS (rope_theta=%f)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 500000.0, got %f)\n", val);
            failed++;
        }
    }

    /* Test 5: Parse num_hidden_layers */
    print("\nTest 5: Parse num_hidden_layers\n");
    {
        char *json = "{\"num_hidden_layers\": 6, \"num_attention_heads\": 6}";
        int val = json_get_int(json, "num_hidden_layers", 0);
        if (val == 6) {
            print("  Result: PASS (num_hidden_layers=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 6, got %d)\n", val);
            failed++;
        }
    }

    /* Test 6: Parse num_attention_heads */
    print("\nTest 6: Parse num_attention_heads\n");
    {
        char *json = "{\"num_hidden_layers\": 6, \"num_attention_heads\": 6}";
        int val = json_get_int(json, "num_attention_heads", 0);
        if (val == 6) {
            print("  Result: PASS (num_attention_heads=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 6, got %d)\n", val);
            failed++;
        }
    }

    /* Test 7: Parse num_key_value_heads (GQA) */
    print("\nTest 7: Parse num_key_value_heads\n");
    {
        char *json = "{\"num_attention_heads\": 32, \"num_key_value_heads\": 8}";
        int val = json_get_int(json, "num_key_value_heads", 0);
        if (val == 8) {
            print("  Result: PASS (num_key_value_heads=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 8, got %d)\n", val);
            failed++;
        }
    }

    /* Test 8: Parse intermediate_size */
    print("\nTest 8: Parse intermediate_size\n");
    {
        char *json = "{\"hidden_size\": 288, \"intermediate_size\": 768}";
        int val = json_get_int(json, "intermediate_size", 0);
        if (val == 768) {
            print("  Result: PASS (intermediate_size=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 768, got %d)\n", val);
            failed++;
        }
    }

    /* Test 9: Parse max_position_embeddings */
    print("\nTest 9: Parse max_position_embeddings\n");
    {
        char *json = "{\"max_position_embeddings\": 2048}";
        int val = json_get_int(json, "max_position_embeddings", 0);
        if (val == 2048) {
            print("  Result: PASS (max_position_embeddings=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 2048, got %d)\n", val);
            failed++;
        }
    }

    /* Test 10: Missing key returns default */
    print("\nTest 10: Missing key returns default\n");
    {
        char *json = "{\"hidden_size\": 288}";
        int val = json_get_int(json, "nonexistent_key", 42);
        if (val == 42) {
            print("  Result: PASS (default=%d)\n", val);
            passed++;
        } else {
            print("  Result: FAIL (expected 42, got %d)\n", val);
            failed++;
        }
    }

    /* Test 11: Full config.json example */
    print("\nTest 11: Full config.json parsing\n");
    {
        char *json = "{\n"
            "  \"model_type\": \"llama\",\n"
            "  \"hidden_size\": 288,\n"
            "  \"intermediate_size\": 768,\n"
            "  \"num_hidden_layers\": 6,\n"
            "  \"num_attention_heads\": 6,\n"
            "  \"num_key_value_heads\": 6,\n"
            "  \"max_position_embeddings\": 256,\n"
            "  \"rope_theta\": 10000.0\n"
            "}";

        int ok = 1;
        if (!json_model_type_is(json, "llama")) ok = 0;
        if (json_get_int(json, "hidden_size", 0) != 288) ok = 0;
        if (json_get_int(json, "intermediate_size", 0) != 768) ok = 0;
        if (json_get_int(json, "num_hidden_layers", 0) != 6) ok = 0;
        if (json_get_int(json, "num_attention_heads", 0) != 6) ok = 0;
        if (json_get_int(json, "num_key_value_heads", 0) != 6) ok = 0;
        if (json_get_int(json, "max_position_embeddings", 0) != 256) ok = 0;

        float theta = json_get_float(json, "rope_theta", 0.0f);
        float diff = theta - 10000.0f;
        if (diff < 0) diff = -diff;
        if (diff >= 1.0f) ok = 0;

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d config.json parsing tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
