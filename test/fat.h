/* FAT disk operations using mtools */
#ifndef FAT_H
#define FAT_H

/* Create a FAT32 disk image */
int fat_create(const char *path, int size_mb);

/* Copy a file to the FAT disk */
int fat_copy_to(const char *disk_path, const char *src_path, const char *dest_name);

/* Copy a file from the FAT disk to a buffer (returns malloc'd buffer, caller frees) */
char *fat_read_file(const char *disk_path, const char *filename, int *size_out);

/* Delete a file from the FAT disk */
int fat_delete(const char *disk_path, const char *filename);

/* List files on disk (for debugging) */
int fat_list(const char *disk_path);

/* Create a directory on the FAT disk */
int fat_mkdir(const char *disk_path, const char *dirname);

/* Copy a file to a directory on the FAT disk */
int fat_copy_to_dir(const char *disk_path, const char *src_path, const char *dest_dir, const char *dest_name);

#endif /* FAT_H */
