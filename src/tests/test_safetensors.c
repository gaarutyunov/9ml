/*
 * test_safetensors.c - Test safetensors file format parsing
 *
 * Tests JSON header parsing, tensor info extraction, and data reading.
 * Creates a minimal safetensors file for testing.
 */

/* Include implementation directly for testing.
 * Note: safetensors.c includes <u.h> and <libc.h>, so we don't include them again. */
#include "format/safetensors.c"

#define TEST_FILE "/tmp/test.safetensors"

/* Create a minimal safetensors file for testing */
static int
create_test_file(void)
{
    int fd;
    char *header = "{"
        "\"tensor1\":{\"dtype\":\"F32\",\"shape\":[4,4],\"data_offsets\":[0,64]},"
        "\"tensor2\":{\"dtype\":\"F16\",\"shape\":[8],\"data_offsets\":[64,80]},"
        "\"__metadata__\":{\"format\":\"test\"}"
        "}";
    uvlong header_size = strlen(header);
    uchar header_buf[8];
    float f32_data[16];
    ushort f16_data[8];
    int i;

    /* Prepare header size (little-endian) */
    header_buf[0] = header_size & 0xff;
    header_buf[1] = (header_size >> 8) & 0xff;
    header_buf[2] = (header_size >> 16) & 0xff;
    header_buf[3] = (header_size >> 24) & 0xff;
    header_buf[4] = (header_size >> 32) & 0xff;
    header_buf[5] = (header_size >> 40) & 0xff;
    header_buf[6] = (header_size >> 48) & 0xff;
    header_buf[7] = (header_size >> 56) & 0xff;

    /* Prepare tensor data */
    for (i = 0; i < 16; i++)
        f32_data[i] = (float)i * 0.5f;

    for (i = 0; i < 8; i++) {
        /* Simple FP16 encoding for small positive integers */
        float val = (float)i;
        /* Approximate FP16: sign=0, exp=15+log2(val), mant=... */
        /* For simplicity, use 1.0, 2.0, etc which have exact representations */
        if (i == 0)
            f16_data[i] = 0x0000;  /* 0.0 */
        else if (i == 1)
            f16_data[i] = 0x3c00;  /* 1.0 */
        else if (i == 2)
            f16_data[i] = 0x4000;  /* 2.0 */
        else if (i == 3)
            f16_data[i] = 0x4200;  /* 3.0 */
        else if (i == 4)
            f16_data[i] = 0x4400;  /* 4.0 */
        else if (i == 5)
            f16_data[i] = 0x4500;  /* 5.0 */
        else if (i == 6)
            f16_data[i] = 0x4600;  /* 6.0 */
        else if (i == 7)
            f16_data[i] = 0x4700;  /* 7.0 */
    }

    /* Write file */
    fd = create(TEST_FILE, OWRITE, 0644);
    if (fd < 0)
        return -1;

    write(fd, header_buf, 8);
    write(fd, header, header_size);
    write(fd, f32_data, sizeof(f32_data));
    write(fd, f16_data, sizeof(f16_data));

    close(fd);
    return 0;
}

/* Helper: compare floats with epsilon */
static int
float_eq(float a, float b, float eps)
{
    float diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < eps;
}

void
main(int argc, char *argv[])
{
    STFile sf;
    STTensor *t;
    int passed = 0;
    int failed = 0;
    float out[32];
    vlong nread;
    int i;

    USED(argc);
    USED(argv);

    print("=== Safetensors Parse Test ===\n");

    /* Create test file */
    if (create_test_file() < 0) {
        print("FAIL: Could not create test file\n");
        exits("fail");
    }

    /* Test 1: Open safetensors file */
    print("\nTest 1: Open safetensors file\n");
    if (st_open(&sf, TEST_FILE) == 0) {
        print("  Result: PASS\n");
        print("  Header size: %ulld\n", sf.header_size);
        print("  Data offset: %ulld\n", sf.data_offset);
        print("  Tensor count: %d\n", sf.n_tensors);
        passed++;
    } else {
        print("  Result: FAIL (could not open file)\n");
        failed++;
        remove(TEST_FILE);
        exits("fail");
    }

    /* Test 2: Find tensor by name */
    print("\nTest 2: Find tensor by name\n");
    t = st_find_tensor(&sf, "tensor1");
    if (t != nil) {
        int ok = 1;
        if (t->dtype != ST_DTYPE_F32) {
            print("  dtype mismatch: %d\n", t->dtype);
            ok = 0;
        }
        if (t->n_dims != 2 || t->dims[0] != 4 || t->dims[1] != 4) {
            print("  shape mismatch: [%ulld, %ulld]\n", t->dims[0], t->dims[1]);
            ok = 0;
        }
        if (ok) {
            print("  Result: PASS\n");
            print("  Name: %s\n", t->name);
            print("  Dtype: %s\n", st_dtype_name(t->dtype));
            print("  Shape: [%ulld, %ulld]\n", t->dims[0], t->dims[1]);
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    } else {
        print("  Result: FAIL (tensor not found)\n");
        failed++;
    }

    /* Test 3: Find F16 tensor */
    print("\nTest 3: Find F16 tensor\n");
    t = st_find_tensor(&sf, "tensor2");
    if (t != nil && t->dtype == ST_DTYPE_F16) {
        print("  Result: PASS\n");
        print("  Dtype: %s\n", st_dtype_name(t->dtype));
        print("  Shape: [%ulld]\n", t->dims[0]);
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    /* Test 4: Calculate tensor elements */
    print("\nTest 4: Calculate tensor elements\n");
    t = st_find_tensor(&sf, "tensor1");
    if (t != nil) {
        uvlong n = st_tensor_nelements(t);
        if (n == 16) {
            print("  Result: PASS (%ulld elements)\n", n);
            passed++;
        } else {
            print("  Result: FAIL (expected 16, got %ulld)\n", n);
            failed++;
        }
    } else {
        print("  Result: FAIL (tensor not found)\n");
        failed++;
    }

    /* Test 5: Read F32 tensor data */
    print("\nTest 5: Read F32 tensor data\n");
    t = st_find_tensor(&sf, "tensor1");
    if (t != nil) {
        nread = st_read_tensor(&sf, t, out, 16);
        if (nread == 16) {
            int ok = 1;
            for (i = 0; i < 16; i++) {
                float expected = (float)i * 0.5f;
                if (!float_eq(out[i], expected, 0.001f)) {
                    print("  out[%d] mismatch: expected %f, got %f\n",
                          i, expected, out[i]);
                    ok = 0;
                }
            }
            if (ok) {
                print("  Result: PASS\n");
                print("  First values: %f, %f, %f, %f\n",
                      out[0], out[1], out[2], out[3]);
                passed++;
            } else {
                print("  Result: FAIL\n");
                failed++;
            }
        } else {
            print("  Result: FAIL (read %lld floats, expected 16)\n", nread);
            failed++;
        }
    } else {
        print("  Result: FAIL (tensor not found)\n");
        failed++;
    }

    /* Test 6: Read F16 tensor data (with conversion) */
    print("\nTest 6: Read F16 tensor data\n");
    t = st_find_tensor(&sf, "tensor2");
    if (t != nil) {
        nread = st_read_tensor(&sf, t, out, 8);
        if (nread == 8) {
            int ok = 1;
            for (i = 0; i < 8; i++) {
                float expected = (float)i;
                if (!float_eq(out[i], expected, 0.01f)) {
                    print("  out[%d] mismatch: expected %f, got %f\n",
                          i, expected, out[i]);
                    ok = 0;
                }
            }
            if (ok) {
                print("  Result: PASS\n");
                print("  Converted values: %f, %f, %f, %f\n",
                      out[0], out[1], out[2], out[3]);
                passed++;
            } else {
                print("  Result: FAIL\n");
                failed++;
            }
        } else {
            print("  Result: FAIL (read %lld floats, expected 8)\n", nread);
            failed++;
        }
    } else {
        print("  Result: FAIL (tensor not found)\n");
        failed++;
    }

    /* Test 7: Find tensors by pattern */
    print("\nTest 7: Find tensors by pattern\n");
    {
        STTensor *matches = st_find_tensors(&sf, "tensor");
        int count = 0;
        for (t = matches; t != nil; t = t->next)
            count++;

        if (count == 2) {
            print("  Result: PASS (found %d tensors)\n", count);
            passed++;
        } else {
            print("  Result: FAIL (expected 2, found %d)\n", count);
            failed++;
        }
        st_tensor_list_free(matches);
    }

    /* Test 8: Dtype size */
    print("\nTest 8: Dtype sizes\n");
    {
        int ok = 1;
        if (st_dtype_size(ST_DTYPE_F32) != 4) {
            print("  F32 size wrong: %d\n", st_dtype_size(ST_DTYPE_F32));
            ok = 0;
        }
        if (st_dtype_size(ST_DTYPE_F16) != 2) {
            print("  F16 size wrong: %d\n", st_dtype_size(ST_DTYPE_F16));
            ok = 0;
        }
        if (st_dtype_size(ST_DTYPE_I8) != 1) {
            print("  I8 size wrong: %d\n", st_dtype_size(ST_DTYPE_I8));
            ok = 0;
        }
        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    }

    /* Cleanup */
    st_close(&sf);
    remove(TEST_FILE);

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d safetensors tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
