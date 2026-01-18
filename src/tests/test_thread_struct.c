/* Test threading with channels in struct (like model.c) */
#include <u.h>
#include <libc.h>
#include <thread.h>

#define STACK 8192

/* WorkItem like in model.c */
typedef struct WorkItem WorkItem;
struct WorkItem {
    void (*fn)(void*);
    void *arg;
    WorkItem *next;
};

/* ThreadPool like in model.c */
typedef struct ThreadPool ThreadPool;
struct ThreadPool {
    int nworkers;
    int active;
    Channel *work;
    Channel *done;
    Channel *shutdown;
    Channel *ready;
};

static void
worker_proc(void *arg)
{
    ThreadPool *p = arg;
    WorkItem *item;

    fprint(2, "worker_proc: started, p=%p\n", p);

    /* Signal that we're ready */
    fprint(2, "worker_proc: signaling ready\n");
    sendp(p->ready, (void*)1);
    fprint(2, "worker_proc: ready sent, entering main loop\n");

    for (;;) {
        item = recvp(p->work);
        if (item == nil) {
            fprint(2, "worker_proc: got nil, shutting down\n");
            sendp(p->shutdown, (void*)1);
            break;
        }

        fprint(2, "worker_proc: got work item %p\n", item);

        if (item->fn != nil) {
            item->fn(item->arg);
        }

        sendp(p->done, (void*)1);
        free(item);
    }

    threadexits(nil);
}

static void
test_work_fn(void *arg)
{
    int val = (int)(uintptr)arg;
    print("work function called with value: %d\n", val);
}

void
threadmain(int argc, char *argv[])
{
    int i;
    int nworkers = 2;
    ThreadPool *p;

    USED(argc);
    USED(argv);

    print("=== Thread Test with Struct (like model.c) ===\n");

    fprint(2, "Allocating ThreadPool\n");
    p = malloc(sizeof(ThreadPool));
    if (p == nil) {
        fprint(2, "malloc failed\n");
        threadexits("malloc");
    }

    p->nworkers = nworkers;
    p->active = 1;

    fprint(2, "Creating channels\n");
    p->work = chancreate(sizeof(void*), nworkers * 4);
    p->done = chancreate(sizeof(void*), nworkers * 4);
    p->shutdown = chancreate(sizeof(void*), nworkers);
    p->ready = chancreate(sizeof(void*), nworkers);

    fprint(2, "Channels: work=%p done=%p shutdown=%p ready=%p\n",
           p->work, p->done, p->shutdown, p->ready);

    if (p->work == nil || p->done == nil || p->shutdown == nil || p->ready == nil) {
        fprint(2, "chancreate failed\n");
        threadexits("chancreate");
    }

    /* Create workers */
    fprint(2, "Creating %d workers\n", nworkers);
    for (i = 0; i < nworkers; i++) {
        fprint(2, "Creating worker %d\n", i);
        proccreate(worker_proc, p, STACK);
    }

    /* Wait for workers to be ready */
    fprint(2, "Waiting for workers to be ready\n");
    for (i = 0; i < nworkers; i++) {
        recvp(p->ready);
        fprint(2, "Worker %d ready\n", i);
    }

    print("Pool created, submitting work\n");

    /* Submit some work */
    for (i = 1; i <= 4; i++) {
        WorkItem *item = malloc(sizeof(WorkItem));
        if (item == nil) {
            fprint(2, "malloc work item failed\n");
            continue;
        }
        item->fn = test_work_fn;
        item->arg = (void*)(uintptr)i;
        item->next = nil;
        fprint(2, "Sending work item %d\n", i);
        sendp(p->work, item);
    }

    /* Wait for work to complete */
    fprint(2, "Waiting for 4 work items to complete\n");
    for (i = 0; i < 4; i++) {
        recvp(p->done);
        fprint(2, "Work item %d completed\n", i+1);
    }

    print("Work complete, shutting down\n");

    /* Shutdown workers */
    for (i = 0; i < nworkers; i++) {
        sendp(p->work, nil);
    }

    /* Wait for shutdown acks */
    for (i = 0; i < nworkers; i++) {
        recvp(p->shutdown);
    }

    chanfree(p->work);
    chanfree(p->done);
    chanfree(p->shutdown);
    chanfree(p->ready);
    free(p);

    print("test complete\n");
    threadexits(0);
}
