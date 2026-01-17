/* QEMU VM management */
#ifndef QEMU_H
#define QEMU_H

#include <sys/types.h>

typedef struct {
    pid_t pid;
    int stdin_fd;       /* Write to send keyboard input */
    int stdout_fd;      /* Read serial output */
    const char *disk_image;
    const char *shared_image;
} QemuVM;

/* Download 9front.qcow2 if not present (uses curl) */
int qemu_ensure_disk(const char *disk_path);

/* Start QEMU with 9front */
int qemu_start(QemuVM *vm, const char *disk_image, const char *shared_image);

/* Send a line of text to the VM (simulates keyboard input) */
int qemu_send(QemuVM *vm, const char *text);

/* Send a line and newline */
int qemu_sendln(QemuVM *vm, const char *text);

/* Wait for output containing a pattern (with timeout in seconds) */
int qemu_wait_for(QemuVM *vm, const char *pattern, int timeout_secs);

/* Sleep helper */
void qemu_sleep(int seconds);

/* Shutdown the VM */
int qemu_shutdown(QemuVM *vm);

/* Kill any lingering QEMU processes */
void qemu_killall(void);

#endif /* QEMU_H */
