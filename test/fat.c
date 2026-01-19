/* FAT disk operations using mtools */
#include "fat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int fat_create(const char *path, int size_mb) {
    char cmd[1024];

    /* Remove existing file */
    unlink(path);

    /* Create image file with truncate (faster than dd) */
    snprintf(cmd, sizeof(cmd), "truncate -s %dM %s", size_mb, path);
    if (system(cmd) != 0) {
        fprintf(stderr, "fat_create: truncate failed\n");
        return -1;
    }

    /* First zero out the first MB to ensure clean MBR/FAT area */
    snprintf(cmd, sizeof(cmd), "dd if=/dev/zero of=%s bs=1M count=1 conv=notrunc status=none", path);
    system(cmd);  /* Ignore errors */

    /* Format as FAT32 using mformat.
     * -F = FAT32 format
     * Note: Don't specify cluster size as mformat will choose optimal settings.
     */
    snprintf(cmd, sizeof(cmd), "mformat -i %s -F :: 2>/dev/null", path);
    if (system(cmd) != 0) {
        fprintf(stderr, "fat_create: mformat failed\n");
        return -1;
    }

    return 0;
}

int fat_copy_to(const char *disk_path, const char *src_path, const char *dest_name) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mcopy -i %s %s ::%s", disk_path, src_path, dest_name);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "fat_copy_to: mcopy failed for %s\n", src_path);
        return -1;
    }
    return 0;
}

char *fat_read_file(const char *disk_path, const char *filename, int *size_out) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mcopy -i %s ::%s -", disk_path, filename);

    FILE *fp = popen(cmd, "r");
    if (!fp) {
        if (size_out) *size_out = 0;
        return NULL;
    }

    /* Read all output */
    size_t capacity = 4096;
    size_t size = 0;
    char *buf = malloc(capacity);
    if (!buf) {
        pclose(fp);
        if (size_out) *size_out = 0;
        return NULL;
    }

    while (!feof(fp)) {
        if (size + 1024 > capacity) {
            capacity *= 2;
            char *newbuf = realloc(buf, capacity);
            if (!newbuf) {
                free(buf);
                pclose(fp);
                if (size_out) *size_out = 0;
                return NULL;
            }
            buf = newbuf;
        }
        size_t n = fread(buf + size, 1, 1024, fp);
        size += n;
        if (n < 1024) break;
    }

    int status = pclose(fp);
    if (status != 0) {
        free(buf);
        if (size_out) *size_out = 0;
        return NULL;
    }

    /* Null terminate */
    buf[size] = '\0';
    if (size_out) *size_out = (int)size;
    return buf;
}

int fat_delete(const char *disk_path, const char *filename) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mdel -i %s ::%s 2>/dev/null", disk_path, filename);
    system(cmd);  /* Ignore errors - file may not exist */
    return 0;
}

int fat_list(const char *disk_path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mdir -i %s ::", disk_path);
    return system(cmd);
}

int fat_mkdir(const char *disk_path, const char *dirname) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mmd -i %s ::%s 2>/dev/null", disk_path, dirname);
    int ret = system(cmd);
    /* mmd returns error if dir exists, which is ok */
    (void)ret;
    return 0;
}

int fat_copy_to_dir(const char *disk_path, const char *src_path, const char *dest_dir, const char *dest_name) {
    char cmd[1024];
    char dest[512];
    if (dest_dir && dest_dir[0]) {
        snprintf(dest, sizeof(dest), "%s/%s", dest_dir, dest_name);
    } else {
        snprintf(dest, sizeof(dest), "%s", dest_name);
    }
    snprintf(cmd, sizeof(cmd), "mcopy -i %s %s ::%s", disk_path, src_path, dest);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "fat_copy_to_dir: mcopy failed for %s -> %s\n", src_path, dest);
        return -1;
    }
    return 0;
}
