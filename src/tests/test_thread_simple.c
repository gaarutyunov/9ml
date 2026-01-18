/* Simple test for proccreate + channels in 9front */
#include <u.h>
#include <libc.h>
#include <thread.h>

#define STACK 8192

Channel *work;
Channel *done;
int nworkers = 2;

static void
worker(void *arg)
{
	int id = (int)(uintptr)arg;
	void *item;

	print("worker %d started\n", id);

	for(;;) {
		item = recvp(work);
		if(item == nil) {
			print("worker %d shutting down\n", id);
			break;
		}
		print("worker %d got work: %d\n", id, (int)(uintptr)item);
		sendp(done, (void*)1);
	}

	threadexits(nil);
}

void
threadmain(int argc, char *argv[])
{
	int i;

	USED(argc);
	USED(argv);

	print("=== Simple Thread Pool Test ===\n");

	/* Create channels */
	work = chancreate(sizeof(void*), 4);
	done = chancreate(sizeof(void*), 4);

	if(work == nil || done == nil) {
		fprint(2, "chancreate failed\n");
		threadexits("chancreate");
	}

	print("channels created\n");

	/* Create worker procs */
	for(i = 0; i < nworkers; i++) {
		print("creating worker %d\n", i);
		proccreate(worker, (void*)(uintptr)i, STACK);
	}

	print("workers created, sending work...\n");

	/* Send some work */
	for(i = 1; i <= 4; i++) {
		print("sending work item %d\n", i);
		sendp(work, (void*)(uintptr)i);
	}

	/* Wait for work to complete */
	print("waiting for work to complete...\n");
	for(i = 0; i < 4; i++) {
		recvp(done);
		print("work item completed\n");
	}

	/* Shutdown workers */
	print("shutting down workers...\n");
	for(i = 0; i < nworkers; i++) {
		sendp(work, nil);
	}

	print("test complete\n");
	threadexits(0);
}
