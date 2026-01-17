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
               "-cpu", "max",
               "-accel", "tcg",
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
        vm->pid = 0;
    }
    return 0;
}

void qemu_killall(void) {
    system("pkill -9 -f qemu-system 2>/dev/null");
    sleep(1);
}
