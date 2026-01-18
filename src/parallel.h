/* Parallel execution support for Plan 9 using libthread */

/* Thread pool for parallel computation */
typedef struct ThreadPool ThreadPool;

/* Work item for the thread pool */
typedef struct WorkItem WorkItem;
struct WorkItem {
    void (*fn)(void*);
    void *arg;
    WorkItem *next;
};

/* Thread pool structure */
struct ThreadPool {
    int nworkers;           /* number of worker threads */
    int active;             /* pool is active */
    Channel *work;          /* work queue channel */
    Channel *done;          /* completion signal channel */
    Channel *shutdown;      /* shutdown acknowledgment channel */
    Channel *ready;         /* workers signal ready on startup */
    QLock lock;             /* protects shutdown */
};

/* Global configuration */
typedef struct {
    int nthreads;           /* number of threads to use (0 = auto-detect) */
    int use_simd;           /* whether to use SIMD optimizations */
} OptConfig;

extern OptConfig opt_config;

/* Initialize optimization configuration */
void opt_init(void);

/* Detect number of CPUs */
int cpu_count(void);

/* Thread pool management */
ThreadPool* pool_create(int nworkers);
void pool_destroy(ThreadPool *p);
void pool_submit(ThreadPool *p, void (*fn)(void*), void *arg);
void pool_wait(ThreadPool *p, int njobs);

/* Parallel execution helpers */
void parallel_for(ThreadPool *p, int start, int end, void (*fn)(int, void*), void *arg);

/* Worker context for attention head parallelization */
typedef struct {
    int h;              /* head index */
    int head_size;
    int pos;
    int seq_len;
    int kv_dim;
    int kv_mul;
    int loff;
    float *q;
    float *att;
    float *xb;
    float *key_cache;
    float *value_cache;
    void (*softmax_fn)(float*, int);
} HeadContext;

/* Parallel attention worker */
void attention_head_worker(void *arg);
