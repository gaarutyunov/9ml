/* QEMU VM management */
#include "qemu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <time.h>

/* 9front download URL */
#define NINEFRONT_URL "https://iso.only9fans.com/release/9front-11194.386.iso.qcow2.bz2"

/* Track PIDs of VMs started by this harness to avoid killing user's VMs */
#define MAX_TRACKED_VMS 16
#define HARNESS_PID_FILE "/tmp/9ml_harness_pids"

static pid_t tracked_pids[MAX_TRACKED_VMS];
static int tracked_count = 0;

/* Write tracked PIDs to file for recovery after crash */
static void save_tracked_pids(void) {
    FILE *f = fopen(HARNESS_PID_FILE, "w");
    if (f) {
        for (int i = 0; i < tracked_count; i++) {
            fprintf(f, "%d\n", tracked_pids[i]);
        }
        fclose(f);
    }
}

static void track_pid(pid_t pid) {
    if (tracked_count < MAX_TRACKED_VMS && pid > 0) {
        tracked_pids[tracked_count++] = pid;
        save_tracked_pids();
    }
}

static void untrack_pid(pid_t pid) {
    for (int i = 0; i < tracked_count; i++) {
        if (tracked_pids[i] == pid) {
            /* Shift remaining elements down */
            for (int j = i; j < tracked_count - 1; j++) {
                tracked_pids[j] = tracked_pids[j + 1];
            }
            tracked_count--;
            save_tracked_pids();
            return;
        }
    }
}

void qemu_tracker_init(void) {
    /* Load any orphaned PIDs from previous crashed run */
    tracked_count = 0;
    FILE *f = fopen(HARNESS_PID_FILE, "r");
    if (f) {
        pid_t pid;
        while (fscanf(f, "%d", &pid) == 1 && tracked_count < MAX_TRACKED_VMS) {
            /* Check if process still exists */
            if (kill(pid, 0) == 0) {
                tracked_pids[tracked_count++] = pid;
            }
        }
        fclose(f);
    }

    /* Kill any orphaned VMs from previous runs */
    if (tracked_count > 0) {
        printf("Found %d orphaned VMs from previous run, cleaning up...\n", tracked_count);
        qemu_killall();
    }

    /* Clear the PID file */
    unlink(HARNESS_PID_FILE);
}

void qemu_tracker_cleanup(void) {
    /* Kill any remaining tracked VMs */
    qemu_killall();
    /* Remove PID file */
    unlink(HARNESS_PID_FILE);
}

int qemu_ensure_disk(const char *disk_path) {
    /* Check if disk already exists */
    if (access(disk_path, F_OK) == 0) {
        return 0;
    }

    printf("Downloading 9front disk image...\n");

    /* Download and decompress */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "curl -fSL '%s' | bunzip2 > '%s'",
             NINEFRONT_URL, disk_path);

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Failed to download 9front image\n");
        unlink(disk_path);
        return -1;
    }

    printf("Download complete.\n");
    return 0;
}

int qemu_start(QemuVM *vm, const char *disk_image, const char *shared_image) {
    int stdin_pipe[2];
    int stdout_pipe[2];

    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0) {
        perror("pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* Child - set up pipes and exec qemu */
        close(stdin_pipe[1]);  /* Close write end of stdin pipe */
        close(stdout_pipe[0]); /* Close read end of stdout pipe */

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stdout_pipe[1], STDERR_FILENO);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        /* Format drive arguments */
        char disk_arg[512], shared_arg[512];
        snprintf(disk_arg, sizeof(disk_arg), "file=%s,format=qcow2,if=virtio", disk_image);
        snprintf(shared_arg, sizeof(shared_arg), "file=%s,format=raw,if=virtio", shared_image);

        execlp("qemu-system-x86_64", "qemu-system-x86_64",
               "-m", "512",
               "-smp", "4",
               "-cpu", "host",
               "-accel", "kvm",
               "-drive", disk_arg,
               "-drive", shared_arg,
               "-display", "none",
               "-serial", "mon:stdio",
               NULL);

        perror("execlp");
        _exit(1);
    }

    /* Parent */
    close(stdin_pipe[0]);  /* Close read end of stdin pipe */
    close(stdout_pipe[1]); /* Close write end of stdout pipe */

    /* Make stdout non-blocking for polling */
    int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
    fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);

    vm->pid = pid;
    vm->stdin_fd = stdin_pipe[1];
    vm->stdout_fd = stdout_pipe[0];
    vm->disk_image = disk_image;
    vm->shared_image = shared_image;
    vm->net_arg = NULL;
    vm->is_cpu = 0;

    /* Track this PID so we only kill VMs we started */
    track_pid(pid);

    return 0;
}

int qemu_start_with_net(QemuVM *vm, const char *disk_image, const char *shared_image,
                        const char *net_type, int is_cpu) {
    int stdin_pipe[2];
    int stdout_pipe[2];

    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0) {
        perror("pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* Child - set up pipes and exec qemu */
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stdout_pipe[1], STDERR_FILENO);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        /* Format drive arguments */
        char disk_arg[512], shared_arg[512];
        snprintf(disk_arg, sizeof(disk_arg), "file=%s,format=qcow2,if=virtio", disk_image);
        /* Disable locking on shared disk so both VMs can access it */
        snprintf(shared_arg, sizeof(shared_arg), "file=%s,format=raw,if=virtio,file.locking=off", shared_image);

        /* Network arguments */
        char nic_arg[256], net_arg[256];
        snprintf(nic_arg, sizeof(nic_arg), "virtio-net-pci,netdev=net0,mac=52:54:00:12:34:%02x",
                 is_cpu ? 1 : 2);
        snprintf(net_arg, sizeof(net_arg), "%s", net_type);

        /* CPU VM gets more memory for model loading */
        const char *mem = is_cpu ? "2048" : "512";

        execlp("qemu-system-x86_64", "qemu-system-x86_64",
               "-m", mem,
               "-smp", "4",
               "-cpu", "host",
               "-accel", "kvm",
               "-drive", disk_arg,
               "-drive", shared_arg,
               "-device", nic_arg,
               "-netdev", net_arg,
               "-display", "none",
               "-serial", "mon:stdio",
               NULL);

        perror("execlp");
        _exit(1);
    }

    /* Parent */
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
    fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);

    vm->pid = pid;
    vm->stdin_fd = stdin_pipe[1];
    vm->stdout_fd = stdout_pipe[0];
    vm->disk_image = disk_image;
    vm->shared_image = shared_image;
    vm->net_arg = strdup(net_type);
    vm->is_cpu = is_cpu;

    /* Track this PID so we only kill VMs we started */
    track_pid(pid);

    return 0;
}

int qemu_send(QemuVM *vm, const char *text) {
    size_t len = strlen(text);
    ssize_t written = write(vm->stdin_fd, text, len);
    if (written < 0) {
        perror("write to qemu");
        return -1;
    }
    return 0;
}

int qemu_sendln(QemuVM *vm, const char *text) {
    if (qemu_send(vm, text) < 0) return -1;
    if (qemu_send(vm, "\n") < 0) return -1;
    return 0;
}

/* Send command and wait for prompt to return */
int qemu_sendln_wait(QemuVM *vm, const char *text, int timeout_secs) {
    if (qemu_sendln(vm, text) < 0) return -1;
    return qemu_wait_for(vm, "term%", timeout_secs);
}

/* Debug flag - set to 1 to see VM output */
static int qemu_debug = 0;

void qemu_set_debug(int enable) {
    qemu_debug = enable;
}

int qemu_wait_for(QemuVM *vm, const char *pattern, int timeout_secs) {
    char buf[4096];
    char linebuf[8192];
    int linepos = 0;
    time_t start = time(NULL);

    while (time(NULL) - start < timeout_secs) {
        struct pollfd pfd = {.fd = vm->stdout_fd, .events = POLLIN};
        int ret = poll(&pfd, 1, 1000);  /* 1 second poll */

        if (ret > 0 && (pfd.revents & POLLIN)) {
            ssize_t n = read(vm->stdout_fd, buf, sizeof(buf) - 1);
            if (n > 0) {
                buf[n] = '\0';

                /* Debug: print VM output */
                if (qemu_debug) {
                    fprintf(stderr, "[VM] %s", buf);
                }

                /* Accumulate into line buffer */
                for (int i = 0; i < n && linepos < (int)sizeof(linebuf) - 1; i++) {
                    linebuf[linepos++] = buf[i];
                }
                linebuf[linepos] = '\0';

                /* Check for pattern */
                if (strstr(linebuf, pattern)) {
                    return 0;
                }

                /* Trim old data if buffer getting full */
                if (linepos > 4096) {
                    memmove(linebuf, linebuf + 2048, linepos - 2048);
                    linepos -= 2048;
                }
            }
        }
    }

    if (qemu_debug) {
        fprintf(stderr, "[VM] TIMEOUT waiting for '%s' after %ds\n", pattern, timeout_secs);
    }

    return -1;  /* Timeout */
}

void qemu_sleep(int seconds) {
    sleep(seconds);
}

int qemu_shutdown(QemuVM *vm) {
    if (vm->pid > 0) {
        /* Send shutdown command */
        qemu_sendln(vm, "fshalt");
        sleep(3);

        /* Close pipes */
        close(vm->stdin_fd);
        close(vm->stdout_fd);

        /* Kill and wait */
        kill(vm->pid, SIGTERM);
        sleep(1);
        kill(vm->pid, SIGKILL);

        int status;
        waitpid(vm->pid, &status, 0);

        /* Untrack this PID */
        untrack_pid(vm->pid);
        vm->pid = 0;
    }
    return 0;
}

void qemu_killall(void) {
    /* Only kill VMs that we started, not user's manual VMs */
    for (int i = 0; i < tracked_count; i++) {
        if (tracked_pids[i] > 0) {
            kill(tracked_pids[i], SIGTERM);
        }
    }
    usleep(500000);  /* Give them time to terminate gracefully */

    for (int i = 0; i < tracked_count; i++) {
        if (tracked_pids[i] > 0) {
            kill(tracked_pids[i], SIGKILL);
            int status;
            waitpid(tracked_pids[i], &status, WNOHANG);
        }
    }
    tracked_count = 0;
}

/* Dual-VM management */

int dualvm_start(DualVM *d, const char *cpu_disk, const char *term_disk, const char *shared_image) {
    memset(d, 0, sizeof(*d));

    printf("Starting CPU VM (server)...\n");

    /* CPU VM listens on socket */
    char cpu_net[256];
    snprintf(cpu_net, sizeof(cpu_net),
             "socket,id=net0,listen=:%d", CPU_VM_NET_PORT);

    if (qemu_start_with_net(&d->cpu, cpu_disk, shared_image, cpu_net, 1) != 0) {
        fprintf(stderr, "Failed to start CPU VM\n");
        return -1;
    }

    /* Give CPU VM time to start listening on socket */
    sleep(5);

    printf("Starting Terminal VM (client)...\n");

    /* Terminal VM connects to CPU */
    char term_net[256];
    snprintf(term_net, sizeof(term_net),
             "socket,id=net0,connect=127.0.0.1:%d", CPU_VM_NET_PORT);

    if (qemu_start_with_net(&d->terminal, term_disk, shared_image, term_net, 0) != 0) {
        fprintf(stderr, "Failed to start Terminal VM\n");
        qemu_shutdown(&d->cpu);
        return -1;
    }

    return 0;
}

int dualvm_shutdown(DualVM *d) {
    qemu_shutdown(&d->terminal);
    qemu_shutdown(&d->cpu);
    return 0;
}

int dualvm_boot_and_mount_shared(DualVM *d) {
    /* Boot CPU VM */
    printf("Waiting for CPU VM bootargs...\n");
    if (qemu_wait_for(&d->cpu, "bootargs", 60) < 0) {
        fprintf(stderr, "Timeout waiting for CPU VM bootargs\n");
        return -1;
    }
    qemu_sendln(&d->cpu, "");
    printf("Waiting for CPU VM user prompt...\n");
    if (qemu_wait_for(&d->cpu, "user", 60) < 0) {
        fprintf(stderr, "Timeout waiting for CPU VM user\n");
        return -1;
    }
    qemu_sendln(&d->cpu, "");
    printf("Waiting for CPU VM shell...\n");
    if (qemu_wait_for(&d->cpu, "term%", 120) < 0) {
        fprintf(stderr, "Timeout waiting for CPU VM shell\n");
        return -1;
    }

    /* Boot Terminal VM */
    printf("Waiting for Terminal VM bootargs...\n");
    if (qemu_wait_for(&d->terminal, "bootargs", 60) < 0) {
        fprintf(stderr, "Timeout waiting for Terminal VM bootargs\n");
        return -1;
    }
    qemu_sendln(&d->terminal, "");
    printf("Waiting for Terminal VM user prompt...\n");
    if (qemu_wait_for(&d->terminal, "user", 60) < 0) {
        fprintf(stderr, "Timeout waiting for Terminal VM user\n");
        return -1;
    }
    qemu_sendln(&d->terminal, "");
    printf("Waiting for Terminal VM shell...\n");
    if (qemu_wait_for(&d->terminal, "term%", 120) < 0) {
        fprintf(stderr, "Timeout waiting for Terminal VM shell\n");
        return -1;
    }

    /* Mount shared disk ONLY on CPU VM (FAT doesn't support concurrent access) */
    printf("Mounting shared disk on CPU VM...\n");
    qemu_sendln_wait(&d->cpu, "dossrv -f /dev/sdG0/data shared", 10);
    qemu_sendln_wait(&d->cpu, "mount -c /srv/shared /mnt/host", 10);
    qemu_sendln_wait(&d->cpu, "cd /mnt/host", 5);

    /* Terminal VM does NOT mount shared disk - will use 9P network instead */
    printf("Terminal VM ready (no shared disk - uses 9P network)...\n");

    return 0;
}

int dualvm_configure_network(DualVM *d) {
    /* Configure IP addresses on both VMs */
    printf("Configuring network on CPU VM...\n");
    qemu_sendln_wait(&d->cpu, "ip/ipconfig ether /net/ether0 10.0.0.2 255.255.255.0", 10);

    printf("Configuring network on Terminal VM...\n");
    qemu_sendln_wait(&d->terminal, "ip/ipconfig ether /net/ether0 10.0.0.3 255.255.255.0", 10);

    return 0;
}

/* Start VM with internet access using QEMU user-mode networking (SLIRP/NAT) */
int qemu_start_with_internet(QemuVM *vm, const char *disk_image, const char *shared_image) {
    int stdin_pipe[2];
    int stdout_pipe[2];

    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0) {
        perror("pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* Child - set up pipes and exec qemu */
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stdout_pipe[1], STDERR_FILENO);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        /* Format drive arguments */
        char disk_arg[512], shared_arg[512];
        snprintf(disk_arg, sizeof(disk_arg), "file=%s,format=qcow2,if=virtio", disk_image);
        snprintf(shared_arg, sizeof(shared_arg), "file=%s,format=raw,if=virtio", shared_image);

        /* User-mode networking with NAT for internet access
         * VM gets 10.0.2.15 via DHCP, gateway at 10.0.2.2, DNS at 10.0.2.3 */
        execlp("qemu-system-x86_64", "qemu-system-x86_64",
               "-m", "2048",
               "-smp", "4",
               "-cpu", "host",
               "-accel", "kvm",
               "-drive", disk_arg,
               "-drive", shared_arg,
               "-device", "virtio-net-pci,netdev=net0",
               "-netdev", "user,id=net0",
               "-display", "none",
               "-serial", "mon:stdio",
               NULL);

        perror("execlp");
        _exit(1);
    }

    /* Parent */
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
    fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);

    vm->pid = pid;
    vm->stdin_fd = stdin_pipe[1];
    vm->stdout_fd = stdout_pipe[0];
    vm->disk_image = disk_image;
    vm->shared_image = shared_image;
    vm->net_arg = NULL;
    vm->is_cpu = 0;

    track_pid(pid);

    return 0;
}

/* Configure Plan 9 networking for internet (QEMU user-mode SLIRP) */
int qemu_configure_internet(QemuVM *vm) {
    /* QEMU user-mode networking provides DHCP, but Plan 9 needs manual config.
     * The SLIRP network is:
     *   VM IP: 10.0.2.15 (or assigned by DHCP)
     *   Gateway: 10.0.2.2
     *   DNS: 10.0.2.3
     */

    /* Configure network interface with DHCP-like static config */
    printf("Configuring internet access...\n");
    qemu_sendln_wait(vm, "ip/ipconfig ether /net/ether0 10.0.2.15 255.255.255.0", 10);

    /* Add default route through gateway */
    qemu_sendln_wait(vm, "echo 'add 0.0.0.0 0.0.0.0 10.0.2.2' > /net/iproute", 5);

    /* Configure DNS properly in ndb format */
    /* First create a proper ndb entry for DNS server */
    qemu_sendln_wait(vm, "echo 'ip=10.0.2.15 sys=term dom=localdomain' > /tmp/net.ndb", 5);
    qemu_sendln_wait(vm, "echo 'dns=10.0.2.3' >> /tmp/net.ndb", 5);

    /* Bind the ndb to /lib/ndb/local */
    qemu_sendln_wait(vm, "cat /tmp/net.ndb >> /lib/ndb/local", 5);

    /* Add https service mapping in proper NDB tuple format */
    /* The format is: protocol=service port=number */
    qemu_sendln_wait(vm, "grep -s https /lib/ndb/common || echo 'tcp=https port=443' >> /lib/ndb/local", 5);

    /* Kill and restart cs (connection server) to pick up changes */
    qemu_sendln_wait(vm, "kill ndb/cs | rc", 5);
    qemu_sendln_wait(vm, "sleep 1", 3);
    qemu_sendln_wait(vm, "ndb/cs", 5);

    /* Verify cs is running and can translate https */
    qemu_sendln_wait(vm, "ndb/csquery tcp!huggingface.co!https > /tmp/cs_test.out >[2=1]", 10);
    qemu_sendln_wait(vm, "cat /tmp/cs_test.out", 5);

    /* Start DNS resolver */
    qemu_sendln_wait(vm, "ndb/dns -r", 5);

    /* Wait for services to start */
    qemu_sleep(3);

    /* Test DNS resolution */
    printf("Testing DNS resolution...\n");
    qemu_sendln_wait(vm, "ndb/dnsquery 10.0.2.3 huggingface.co ip > /tmp/dns_test.out", 10);
    qemu_sendln_wait(vm, "cat /tmp/dns_test.out", 5);

    /* Start webfs for HTTPS support */
    printf("Starting webfs for HTTPS...\n");
    qemu_sendln_wait(vm, "webfs", 5);
    qemu_sleep(2);

    /* Verify webfs is running */
    qemu_sendln_wait(vm, "ls /mnt/web/clone && echo 'webfs OK'", 5);

    return 0;
}
