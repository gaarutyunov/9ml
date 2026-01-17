/*
 * Model Export Tool for 9ml
 * Converts models to binary format expected by run.c/runq.c
 *
 * Works on both Linux and Plan 9
 *
 * Usage:
 *   export info model.bin           - Show model info
 *   export convert in.safetensors out.bin - Convert safetensors to bin
 *   export quantize model.bin model_q80.bin - Quantize model
 */

#ifdef __plan9__
#include <u.h>
#include <libc.h>
#define STDERR 2
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#define nil NULL
#define print printf
#define fprint fprintf
#define STDERR stderr
#define vlong int64_t
#define uvlong uint64_t
#define uchar unsigned char
#define schar signed char
#define OREAD 0
static void exits(const char *s) { exit(s ? 1 : 0); }
#endif

/* Model config (28 bytes in file) */
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

/* Read entire file into memory */
static char *read_file(const char *path, vlong *size_out) {
#ifdef __plan9__
    int fd;
    Dir *d;
    char *buf;
    vlong n, pos;

    fd = open(path, OREAD);
    if (fd < 0) {
        fprint(STDERR, "Cannot open %s\n", path);
        return nil;
    }
    d = dirfstat(fd);
    if (d == nil) {
        close(fd);
        return nil;
    }
    *size_out = d->length;
    free(d);

    buf = malloc(*size_out);
    if (!buf) {
        close(fd);
        return nil;
    }

    pos = 0;
    while (pos < *size_out) {
        n = read(fd, buf + pos, (*size_out - pos) > 8192 ? 8192 : *size_out - pos);
        if (n <= 0) break;
        pos += n;
    }
    close(fd);
    return buf;
#else
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprint(STDERR, "Cannot open %s\n", path);
        return nil;
    }
    fseek(f, 0, SEEK_END);
    *size_out = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = malloc(*size_out);
    if (!buf) {
        fclose(f);
        return nil;
    }
    if (fread(buf, 1, *size_out, f) != (size_t)*size_out) {
        free(buf);
        fclose(f);
        return nil;
    }
    fclose(f);
    return buf;
#endif
}

/* Write buffer to file */
static int write_file(const char *path, const void *data, vlong size) {
#ifdef __plan9__
    int fd;
    vlong written = 0;

    fd = create(path, OWRITE, 0644);
    if (fd < 0) {
        fprint(STDERR, "Cannot create %s\n", path);
        return -1;
    }

    while (written < size) {
        long chunk = (size - written) > 8192 ? 8192 : size - written;
        long n = write(fd, (char*)data + written, chunk);
        if (n <= 0) {
            close(fd);
            return -1;
        }
        written += n;
    }
    close(fd);
    return 0;
#else
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprint(STDERR, "Cannot create %s\n", path);
        return -1;
    }
    if (fwrite(data, 1, size, f) != (size_t)size) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
#endif
}

/* Show model info */
static int cmd_info(const char *path) {
    vlong size;
    char *data = read_file(path, &size);
    if (!data) return 1;

    /* Check for quantized format (magic number) */
    unsigned int magic = *(unsigned int*)data;
    int quantized = (magic == 0x616b3432);  /* "ak42" */

    Config c;
    float *weights;
    int group_size = 0;

    if (quantized) {
        /* Version 2 quantized format */
        int version = *(int*)(data + 4);
        if (version != 2) {
            fprint(STDERR, "Unknown quantized version: %d\n", version);
            free(data);
            return 1;
        }

        int *config_ptr = (int*)(data + 8);
        c.dim = config_ptr[0];
        c.hidden_dim = config_ptr[1];
        c.n_layers = config_ptr[2];
        c.n_heads = config_ptr[3];
        c.n_kv_heads = config_ptr[4];
        c.vocab_size = config_ptr[5];
        c.seq_len = config_ptr[6];

        uchar shared_classifier = *(uchar*)(data + 36);
        group_size = *(int*)(data + 37);

        print("Quantized Model (Q8_0)\n");
        print("  Version: %d\n", version);
        print("  Shared classifier: %s\n", shared_classifier ? "yes" : "no");
        print("  Group size: %d\n", group_size);
    } else {
        /* FP32 format */
        int *config_ptr = (int*)data;
        c.dim = config_ptr[0];
        c.hidden_dim = config_ptr[1];
        c.n_layers = config_ptr[2];
        c.n_heads = config_ptr[3];
        c.n_kv_heads = config_ptr[4];
        c.vocab_size = config_ptr[5];
        c.seq_len = config_ptr[6];

        int shared_weights = c.vocab_size > 0 ? 1 : 0;
        if (c.vocab_size < 0) c.vocab_size = -c.vocab_size;

        weights = (float*)(data + 28);

        print("FP32 Model\n");
        print("  Shared weights: %s\n", shared_weights ? "yes" : "no");
    }

    print("\nConfig:\n");
    print("  dim:        %d\n", c.dim);
    print("  hidden_dim: %d\n", c.hidden_dim);
    print("  n_layers:   %d\n", c.n_layers);
    print("  n_heads:    %d\n", c.n_heads);
    print("  n_kv_heads: %d\n", c.n_kv_heads);
    print("  vocab_size: %d\n", c.vocab_size);
    print("  seq_len:    %d\n", c.seq_len);
    print("\nFile size: %lld bytes\n", (long long)size);

    if (!quantized) {
        /* Calculate expected size */
        int head_size = c.dim / c.n_heads;
        uvlong n_layers = c.n_layers;
        uvlong expected = 28;  /* config */
        expected += c.vocab_size * c.dim * sizeof(float);  /* token_embedding */
        expected += n_layers * c.dim * sizeof(float);  /* rms_att_weight */
        expected += n_layers * c.dim * c.n_heads * head_size * sizeof(float);  /* wq */
        expected += n_layers * c.dim * c.n_kv_heads * head_size * sizeof(float);  /* wk */
        expected += n_layers * c.dim * c.n_kv_heads * head_size * sizeof(float);  /* wv */
        expected += n_layers * c.n_heads * head_size * c.dim * sizeof(float);  /* wo */
        expected += n_layers * c.dim * sizeof(float);  /* rms_ffn_weight */
        expected += n_layers * c.dim * c.hidden_dim * sizeof(float);  /* w1 */
        expected += n_layers * c.hidden_dim * c.dim * sizeof(float);  /* w2 */
        expected += n_layers * c.dim * c.hidden_dim * sizeof(float);  /* w3 */
        expected += c.dim * sizeof(float);  /* rms_final_weight */
        expected += c.seq_len * head_size / 2 * sizeof(float);  /* freq_cis_real */
        expected += c.seq_len * head_size / 2 * sizeof(float);  /* freq_cis_imag */
        /* wcls may or may not be included depending on shared_weights */

        print("Expected min size: %llu bytes\n", (unsigned long long)expected);
    }

    free(data);
    return 0;
}

/* Quantize a float array to int8 with scaling */
static void quantize(schar *q, float *s, const float *x, int n, int gs) {
    int num_groups = n / gs;
    const float Q_MAX = 127.0f;

    for (int g = 0; g < num_groups; g++) {
        float wmax = 0.0f;
        for (int i = 0; i < gs; i++) {
            float val = x[g * gs + i];
            if (val < 0) val = -val;
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        s[g] = scale;
        for (int i = 0; i < gs; i++) {
            float quant = (scale > 0) ? x[g * gs + i] / scale : 0;
#ifdef __plan9__
            q[g * gs + i] = (schar)(quant + 0.5f - (quant < 0 ? 1 : 0));
#else
            q[g * gs + i] = (schar)roundf(quant);
#endif
        }
    }
}

/* Quantize model */
static int cmd_quantize(const char *in_path, const char *out_path) {
    const int GS = 32;  /* Group size */

    vlong size;
    char *data = read_file(in_path, &size);
    if (!data) return 1;

    /* Check if already quantized */
    unsigned int magic = *(unsigned int*)data;
    if (magic == 0x616b3432) {
        fprint(STDERR, "Model is already quantized\n");
        free(data);
        return 1;
    }

    /* Read config */
    int *config_ptr = (int*)data;
    Config c;
    c.dim = config_ptr[0];
    c.hidden_dim = config_ptr[1];
    c.n_layers = config_ptr[2];
    c.n_heads = config_ptr[3];
    c.n_kv_heads = config_ptr[4];
    c.vocab_size = config_ptr[5];
    c.seq_len = config_ptr[6];

    uchar shared_weights = c.vocab_size > 0 ? 1 : 0;
    if (c.vocab_size < 0) c.vocab_size = -c.vocab_size;

    int head_size = c.dim / c.n_heads;
    float *weights = (float*)(data + 28);

    print("Quantizing model (GS=%d)...\n", GS);

    /* Calculate sizes */
    uvlong n_layers = c.n_layers;
    uvlong tok_emb_size = c.vocab_size * c.dim;
    uvlong wq_size = n_layers * c.dim * c.n_heads * head_size;
    uvlong wk_size = n_layers * c.dim * c.n_kv_heads * head_size;
    uvlong wv_size = n_layers * c.dim * c.n_kv_heads * head_size;
    uvlong wo_size = n_layers * c.n_heads * head_size * c.dim;
    uvlong w1_size = n_layers * c.dim * c.hidden_dim;
    uvlong w2_size = n_layers * c.hidden_dim * c.dim;
    uvlong w3_size = n_layers * c.dim * c.hidden_dim;
    uvlong wcls_size = shared_weights ? 0 : c.dim * c.vocab_size;

    /* Calculate output size */
    uvlong header_size = 256;  /* Fixed header */
    uvlong fp32_size = (n_layers * c.dim + n_layers * c.dim + c.dim) * sizeof(float);  /* rmsnorms */

    uvlong quant_size = 0;
    quant_size += tok_emb_size + tok_emb_size / GS * sizeof(float);
    quant_size += wq_size + wq_size / GS * sizeof(float);
    quant_size += wk_size + wk_size / GS * sizeof(float);
    quant_size += wv_size + wv_size / GS * sizeof(float);
    quant_size += wo_size + wo_size / GS * sizeof(float);
    quant_size += w1_size + w1_size / GS * sizeof(float);
    quant_size += w2_size + w2_size / GS * sizeof(float);
    quant_size += w3_size + w3_size / GS * sizeof(float);
    if (!shared_weights) {
        quant_size += wcls_size + wcls_size / GS * sizeof(float);
    }

    uvlong out_size = header_size + fp32_size + quant_size;
    char *out = calloc(1, out_size);
    if (!out) {
        fprint(STDERR, "Out of memory\n");
        free(data);
        return 1;
    }

    /* Write header */
    *(unsigned int*)out = 0x616b3432;  /* magic "ak42" */
    *(int*)(out + 4) = 2;  /* version */
    memcpy(out + 8, config_ptr, 28);
    if (c.vocab_size > 0) {
        *(int*)(out + 8 + 20) = c.vocab_size;  /* ensure positive vocab_size in output */
    }
    *(uchar*)(out + 36) = shared_weights;
    *(int*)(out + 37) = GS;

    /* Write weights */
    char *outp = out + header_size;
    float *inp = weights;

    /* Copy fp32 rmsnorm weights */
    uvlong rms_att_size = n_layers * c.dim;
    uvlong rms_ffn_size = n_layers * c.dim;
    uvlong rms_final_size = c.dim;

    /* Skip token embedding for now, copy rms weights first */
    float *tok_emb = inp;
    inp += tok_emb_size;

    float *rms_att = inp;
    inp += rms_att_size;

    /* Skip attention weights */
    float *wq = inp;
    inp += wq_size;
    float *wk = inp;
    inp += wk_size;
    float *wv = inp;
    inp += wv_size;
    float *wo = inp;
    inp += wo_size;

    float *rms_ffn = inp;
    inp += rms_ffn_size;

    /* Skip ffn weights */
    float *w1 = inp;
    inp += w1_size;
    float *w2 = inp;
    inp += w2_size;
    float *w3 = inp;
    inp += w3_size;

    float *rms_final = inp;
    inp += rms_final_size;

    /* freq_cis skipped */
    inp += c.seq_len * head_size / 2 * 2;

    float *wcls = shared_weights ? tok_emb : inp;

    /* Write rmsnorm weights as fp32 */
    memcpy(outp, rms_att, rms_att_size * sizeof(float));
    outp += rms_att_size * sizeof(float);
    memcpy(outp, rms_ffn, rms_ffn_size * sizeof(float));
    outp += rms_ffn_size * sizeof(float);
    memcpy(outp, rms_final, rms_final_size * sizeof(float));
    outp += rms_final_size * sizeof(float);

    /* Write quantized weights */
#define WRITE_QUANT(src, n) do { \
    schar *q = (schar*)outp; \
    float *s = (float*)(outp + (n)); \
    quantize(q, s, src, n, GS); \
    outp += (n) + ((n) / GS) * sizeof(float); \
} while(0)

    print("  Quantizing token embeddings...\n");
    WRITE_QUANT(tok_emb, tok_emb_size);

    print("  Quantizing attention weights...\n");
    WRITE_QUANT(wq, wq_size);
    WRITE_QUANT(wk, wk_size);
    WRITE_QUANT(wv, wv_size);
    WRITE_QUANT(wo, wo_size);

    print("  Quantizing FFN weights...\n");
    WRITE_QUANT(w1, w1_size);
    WRITE_QUANT(w2, w2_size);
    WRITE_QUANT(w3, w3_size);

    if (!shared_weights) {
        print("  Quantizing classifier...\n");
        WRITE_QUANT(wcls, wcls_size);
    }

    /* Write output */
    print("Writing %s...\n", out_path);
    int ret = write_file(out_path, out, out_size);

    free(out);
    free(data);

    if (ret == 0) {
        print("Done. Output size: %llu bytes (%.1f%% of original)\n",
              (unsigned long long)out_size, 100.0 * out_size / size);
    }
    return ret;
}

/* Print usage */
static void usage(void) {
    fprint(STDERR, "Usage: export <command> [args]\n");
    fprint(STDERR, "\nCommands:\n");
    fprint(STDERR, "  info <model.bin>              Show model info\n");
    fprint(STDERR, "  quantize <in.bin> <out.bin>   Quantize FP32 model to Q8_0\n");
    exits("usage");
}

#ifdef __plan9__
void
main(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{
    if (argc < 2) usage();

    if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) usage();
        int ret = cmd_info(argv[2]);
        exits(ret ? "error" : nil);
    }
    else if (strcmp(argv[1], "quantize") == 0) {
        if (argc < 4) usage();
        int ret = cmd_quantize(argv[2], argv[3]);
        exits(ret ? "error" : nil);
    }
    else {
        usage();
    }

#ifndef __plan9__
    return 0;
#endif
}
