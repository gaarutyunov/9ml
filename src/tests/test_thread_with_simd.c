/* Test threading when SIMD assembly is linked but not used */
#include <u.h>
#include <libc.h>
#include <thread.h>

#define STACK 8192

Channel *work;
Channel *done;
int nworkers = 2;

/* External SIMD functions - linked but NOT called */
extern void matmul_simd(float *xout, float *x, float *w, int n, int d);

static void
worker(void *arg)
{
    int id = (int)(uintptr)arg;
    void *item;

    print("worker %d started\n", id);

    for(;;) {
        item = recvp(work);
        if(item == nil)
            break;
        print("worker %d got work: %d\n", id, (int)(uintptr)item);
        sendp(done, (void*)1);
    }

    print("worker %d exiting\n", id);
    threadexits(nil);
}

void
threadmain(int argc, char *argv[])
{
    int i;

    USED(argc);
    USED(argv);

    print("=== Thread Test with SIMD Linked ===\n");

    /* Create channels */
    work = chancreate(sizeof(void*), 4);
    done = chancreate(sizeof(void*), 4);

    if(work == nil || done == nil) {
        fprint(2, "chancreate failed\n");
        threadexits("chancreate");
    }

    print("channels created\n");

    /* Create workers */
    for(i = 0; i < nworkers; i++) {
        print("creating worker %d\n", i);
        proccreate(worker, (void*)(uintptr)i, STACK);
    }

    print("workers created, sending work\n");

    /* Send some work */
    for(i = 1; i <= 4; i++) {
        sendp(work, (void*)(uintptr)i);
    }

    /* Wait for work to complete */
    for(i = 0; i < 4; i++) {
        recvp(done);
    }

    print("work complete, shutting down\n");

    /* Shutdown workers */
    for(i = 0; i < nworkers; i++) {
        sendp(work, nil);
    }

    print("test complete\n");
    threadexits(0);
}
