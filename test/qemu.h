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
    char *net_arg;      /* Network argument (for socket networking) */
    int is_cpu;         /* 1 if this is the CPU (server) VM */
} QemuVM;

/* Dual-VM setup for remote 9P testing */
typedef struct {
    QemuVM cpu;         /* Server VM running llmfs */
    QemuVM terminal;    /* Client VM mounting remote fs */
} DualVM;

/* Network ports for socket networking between VMs */
#define CPU_VM_NET_PORT 10564
#define TERMINAL_VM_NET_PORT 10565

/* Download 9front.qcow2 if not present (uses curl) */
int qemu_ensure_disk(const char *disk_path);

/* Start QEMU with 9front */
int qemu_start(QemuVM *vm, const char *disk_image, const char *shared_image);

/* Start QEMU with socket networking */
int qemu_start_with_net(QemuVM *vm, const char *disk_image, const char *shared_image,
                        const char *net_type, int is_cpu);

/* Send a line of text to the VM (simulates keyboard input) */
int qemu_send(QemuVM *vm, const char *text);

/* Send a line and newline */
int qemu_sendln(QemuVM *vm, const char *text);

/* Send command and wait for prompt to return */
int qemu_sendln_wait(QemuVM *vm, const char *text, int timeout_secs);

/* Wait for output containing a pattern (with timeout in seconds) */
int qemu_wait_for(QemuVM *vm, const char *pattern, int timeout_secs);

/* Enable/disable VM output debugging */
void qemu_set_debug(int enable);

/* Sleep helper */
void qemu_sleep(int seconds);

/* Shutdown the VM */
int qemu_shutdown(QemuVM *vm);

/* Kill only QEMU processes started by this harness */
void qemu_killall(void);

/* Initialize/cleanup VM tracking */
void qemu_tracker_init(void);
void qemu_tracker_cleanup(void);

/* Dual-VM management */
int dualvm_start(DualVM *d, const char *cpu_disk, const char *term_disk, const char *shared_image);
int dualvm_shutdown(DualVM *d);
int dualvm_boot_and_mount_shared(DualVM *d);
int dualvm_configure_network(DualVM *d);

/* Start VM with internet access (user-mode networking with NAT) */
int qemu_start_with_internet(QemuVM *vm, const char *disk_image, const char *shared_image);

/* Configure Plan 9 networking for internet access (DHCP on user-mode network) */
int qemu_configure_internet(QemuVM *vm);

#endif /* QEMU_H */
