/*
 * arch/arch.c - Model architecture registry for 9ml
 *
 * Manages registration and lookup of architecture plugins.
 */

#include <u.h>
#include <libc.h>

#include "arch.h"

/* Maximum number of registered architectures */
#define MAX_ARCHS 16

static ModelArch *registered_archs[MAX_ARCHS];
static int num_archs = 0;

/* Initialize config with default values for extended fields */
void
model_config_init_defaults(ModelConfig *cfg)
{
    cfg->rope_theta = DEFAULT_ROPE_THETA;
    cfg->arch_id = ARCH_UNKNOWN;
}

/* Register an architecture plugin */
void
arch_register(ModelArch *arch)
{
    if (num_archs >= MAX_ARCHS) {
        fprint(2, "arch_register: too many architectures\n");
        return;
    }
    registered_archs[num_archs++] = arch;
}

/* Find architecture by name */
ModelArch *
arch_find(char *name)
{
    int i;
    for (i = 0; i < num_archs; i++) {
        if (strcmp(registered_archs[i]->name, name) == 0) {
            return registered_archs[i];
        }
    }
    return nil;
}

/* Find architecture by ID */
ModelArch *
arch_find_by_id(int arch_id)
{
    int i;
    for (i = 0; i < num_archs; i++) {
        if (registered_archs[i]->arch_id == arch_id) {
            return registered_archs[i];
        }
    }
    return nil;
}

/* Auto-detect architecture from model data */
ModelArch *
arch_detect(void *data, vlong size, ModelConfig *cfg)
{
    int i;
    for (i = 0; i < num_archs; i++) {
        if (registered_archs[i]->detect != nil) {
            if (registered_archs[i]->detect(data, size, cfg)) {
                cfg->arch_id = registered_archs[i]->arch_id;
                return registered_archs[i];
            }
        }
    }
    return nil;
}

/* List registered architectures */
int
arch_list(ModelArch **out, int max)
{
    int i, n;
    n = num_archs < max ? num_archs : max;
    for (i = 0; i < n; i++) {
        out[i] = registered_archs[i];
    }
    return n;
}

/* ----------------------------------------------------------------------------
 * Architecture Initialization
 *
 * Register all built-in architecture plugins. Call this once at startup.
 * ---------------------------------------------------------------------------- */

/* Forward declarations of registration functions */
extern void llama2_register(void);
extern void llama3_register(void);
extern void mistral_register(void);

static int arch_initialized = 0;

void
arch_init(void)
{
    if (arch_initialized) {
        return;
    }

    /* Register built-in architectures in detection order:
     * More specific detectors first (LLaMA 3 checks vocab size),
     * then general ones (LLaMA 2 as default fallback).
     */
    llama3_register();   /* Check for large vocab first */
    mistral_register();  /* Check for sliding window metadata */
    llama2_register();   /* Default fallback for llama2.c format */

    arch_initialized = 1;
}

/* ----------------------------------------------------------------------------
 * Standard RoPE Implementation
 *
 * RoPE (Rotary Position Embedding) rotates query and key vectors using
 * complex-valued rotation based on position. This is the standard
 * implementation used by LLaMA 2, LLaMA 3, and Mistral.
 *
 * The key difference between architectures is rope_theta:
 *   - LLaMA 2: 10000
 *   - LLaMA 3: 500000
 *   - Mistral: 10000 (but with sliding window)
 * ---------------------------------------------------------------------------- */

void
rope_apply_standard(float *q, float *k, int dim, int kv_dim, int pos,
                    int head_size, ModelConfig *cfg)
{
    int i, head_dim, rotn, v;
    float freq, val, fcr, fci, v0, v1;
    float *vec;

    for (i = 0; i < dim; i += 2) {
        head_dim = i % head_size;
        freq = 1.0f / pow(cfg->rope_theta, head_dim / (float)head_size);
        val = pos * freq;
        fcr = cos(val);
        fci = sin(val);

        /* how many vectors? 2 = q & k, 1 = q only (when i >= kv_dim) */
        rotn = i < kv_dim ? 2 : 1;

        for (v = 0; v < rotn; v++) {
            vec = v == 0 ? q : k;  /* the vector to rotate */
            v0 = vec[i];
            v1 = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}
