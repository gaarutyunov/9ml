/* Parallel execution support for Plan 9 using libthread */

#include <u.h>
#include <libc.h>
#include <thread.h>
#include "parallel.h"

/* Global optimization configuration */
OptConfig opt_config = {
    .nthreads = 0,      /* 0 = auto-detect */
    .use_simd = 1,      /* SIMD enabled by default */
};

/* Detect number of CPUs by reading /dev/sysstat */
int
cpu_count(void)
{
    int fd, n, count;
    char *buf, *p;

    /* Use heap instead of stack to avoid stack overflow in threadmain */
    buf = malloc(4096);
    if (buf == nil) {
        return 1;  /* default to 1 if can't detect */
    }

    fd = open("/dev/sysstat", OREAD);
    if (fd < 0) {
        free(buf);
        return 1;
    }

    n = read(fd, buf, 4095);
    close(fd);

    if (n <= 0) {
        free(buf);
        return 1;
    }
    buf[n] = '\0';

    /* Count lines in sysstat - each CPU has one line */
    count = 0;
    for (p = buf; *p; p++) {
        if (*p == '\n') {
            count++;
        }
    }

    free(buf);
    return count > 0 ? count : 1;
}

/* Initialize optimization configuration */
void
opt_init(void)
{
    if (opt_config.nthreads == 0) {
        opt_config.nthreads = cpu_count();
    }
}

/* Worker thread main loop */
static void
worker_proc(void *arg)
{
    ThreadPool *p = arg;
    WorkItem *item;

    /* Signal that we're ready */
    sendp(p->ready, (void*)1);

    for (;;) {
        /* Wait for work item */
        item = recvp(p->work);
        if (item == nil) {
            /* Acknowledge shutdown and exit */
            sendp(p->shutdown, (void*)1);
            break;
        }

        /* Execute the work */
        if (item->fn != nil) {
            item->fn(item->arg);
        }

        /* Signal completion */
        sendp(p->done, (void*)1);

        /* Free the work item */
        free(item);
    }

    threadexits(nil);
}

/* Create a thread pool with nworkers threads */
ThreadPool*
pool_create(int nworkers)
{
    ThreadPool *p;
    int i;

    if (nworkers <= 0) {
        nworkers = cpu_count();
    }

    p = malloc(sizeof(ThreadPool));
    if (p == nil) {
        return nil;
    }

    p->nworkers = nworkers;
    p->active = 1;

    /* Create channels */
    p->work = chancreate(sizeof(void*), nworkers * 4);  /* buffered work queue */
    p->done = chancreate(sizeof(void*), nworkers * 4);  /* buffered done queue */
    p->shutdown = chancreate(sizeof(void*), nworkers);  /* shutdown ack channel */
    p->ready = chancreate(sizeof(void*), nworkers);     /* worker ready signals */

    if (p->work == nil || p->done == nil || p->shutdown == nil || p->ready == nil) {
        if (p->work) chanfree(p->work);
        if (p->done) chanfree(p->done);
        if (p->shutdown) chanfree(p->shutdown);
        if (p->ready) chanfree(p->ready);
        free(p);
        return nil;
    }

    /* Start worker threads (32KB stack) */
    for (i = 0; i < nworkers; i++) {
        proccreate(worker_proc, p, 32768);
    }

    /* Wait for all workers to signal ready */
    for (i = 0; i < nworkers; i++) {
        recvp(p->ready);
    }

    return p;
}

/* Destroy a thread pool */
void
pool_destroy(ThreadPool *p)
{
    int i;

    if (p == nil) {
        return;
    }

    p->active = 0;

    /* Send shutdown signals to all workers */
    for (i = 0; i < p->nworkers; i++) {
        sendp(p->work, nil);
    }

    /* Wait for all workers to acknowledge shutdown */
    for (i = 0; i < p->nworkers; i++) {
        recvp(p->shutdown);
    }

    /* Now safe to free channels */
    chanfree(p->work);
    chanfree(p->done);
    chanfree(p->shutdown);
    chanfree(p->ready);

    free(p);
}

/* Submit work to the thread pool */
void
pool_submit(ThreadPool *p, void (*fn)(void*), void *arg)
{
    WorkItem *item;

    if (p == nil || !p->active) {
        /* No pool - run synchronously */
        if (fn != nil) {
            fn(arg);
        }
        return;
    }

    item = malloc(sizeof(WorkItem));
    if (item == nil) {
        /* Fallback to synchronous execution */
        if (fn != nil) {
            fn(arg);
        }
        return;
    }

    item->fn = fn;
    item->arg = arg;
    item->next = nil;

    sendp(p->work, item);
}

/* Wait for njobs to complete */
void
pool_wait(ThreadPool *p, int njobs)
{
    int i;

    if (p == nil) {
        return;
    }

    for (i = 0; i < njobs; i++) {
        recvp(p->done);
    }
}

/* Parallel for loop context */
typedef struct {
    int idx;
    void (*fn)(int, void*);
    void *arg;
} ParForCtx;

static void
parfor_worker(void *arg)
{
    ParForCtx *ctx = arg;
    ctx->fn(ctx->idx, ctx->arg);
}

/* Execute a parallel for loop */
void
parallel_for(ThreadPool *p, int start, int end, void (*fn)(int, void*), void *arg)
{
    int i, njobs;
    ParForCtx *contexts;

    njobs = end - start;
    if (njobs <= 0) {
        return;
    }

    if (p == nil || p->nworkers <= 1) {
        /* Sequential execution */
        for (i = start; i < end; i++) {
            fn(i, arg);
        }
        return;
    }

    /* Allocate contexts for all jobs */
    contexts = malloc(njobs * sizeof(ParForCtx));
    if (contexts == nil) {
        /* Fallback to sequential */
        for (i = start; i < end; i++) {
            fn(i, arg);
        }
        return;
    }

    /* Submit all jobs */
    for (i = 0; i < njobs; i++) {
        contexts[i].idx = start + i;
        contexts[i].fn = fn;
        contexts[i].arg = arg;
        pool_submit(p, parfor_worker, &contexts[i]);
    }

    /* Wait for completion */
    pool_wait(p, njobs);

    free(contexts);
}

/* Attention head worker - computes attention for a single head */
void
attention_head_worker(void *arg)
{
    HeadContext *ctx = arg;
    int h = ctx->h;
    int head_size = ctx->head_size;
    int pos = ctx->pos;
    int seq_len = ctx->seq_len;
    int kv_dim = ctx->kv_dim;
    int kv_mul = ctx->kv_mul;
    int loff = ctx->loff;

    /* Get pointers for this head */
    float *q = ctx->q + h * head_size;
    float *att = ctx->att + h * seq_len;
    float *xb = ctx->xb + h * head_size;

    int t, i;
    float score, a;
    float *k, *v;

    /* Compute attention scores */
    for (t = 0; t <= pos; t++) {
        k = ctx->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        score = 0.0f;
        for (i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        att[t] = score / sqrt((float)head_size);
    }

    /* Softmax the scores */
    ctx->softmax_fn(att, pos + 1);

    /* Weighted sum of values */
    memset(xb, 0, head_size * sizeof(float));
    for (t = 0; t <= pos; t++) {
        v = ctx->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        a = att[t];
        for (i = 0; i < head_size; i++) {
            xb[i] += a * v[i];
        }
    }
}
