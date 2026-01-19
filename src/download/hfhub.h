/*
 * hfhub.h - HuggingFace Hub API client
 *
 * Supports listing and downloading models from HuggingFace Hub.
 */

#ifndef HFHUB_H
#define HFHUB_H

#include "http.h"

/* HuggingFace Hub API endpoint */
#define HF_API_HOST "huggingface.co"
#define HF_CDN_HOST "cdn-lfs.huggingface.co"

/* Model file info */
typedef struct HFFile HFFile;
struct HFFile {
    char *filename;     /* File name (e.g., "model.safetensors") */
    char *path;         /* Full path in repo */
    vlong size;         /* File size in bytes */
    char *sha256;       /* SHA256 hash (for LFS files) */
    char *lfs_url;      /* Direct download URL for LFS files */
    int is_lfs;         /* 1 if file is LFS-tracked */
    HFFile *next;
};

/* Model info */
typedef struct HFModel HFModel;
struct HFModel {
    char *repo_id;      /* e.g., "meta-llama/Llama-2-7b" */
    char *revision;     /* Branch/tag (default: "main") */
    char *pipeline_tag; /* e.g., "text-generation" */
    char *model_type;   /* e.g., "llama" */
    HFFile *files;      /* List of files in repo */
};

/* Hub client */
typedef struct HFHub HFHub;
struct HFHub {
    HttpClient http;
    char *token;        /* Optional auth token */
    char *cache_dir;    /* Local cache directory */
    char *errmsg;
};

/* Initialize hub client */
void hf_init(HFHub *h, char *cache_dir);

/* Set authentication token */
void hf_set_token(HFHub *h, char *token);

/* Close and free resources */
void hf_close(HFHub *h);

/* Get model info and file list */
int hf_get_model_info(HFHub *h, char *repo_id, char *revision, HFModel *model);

/* Find specific files in model (e.g., GGUF files) */
HFFile *hf_find_files(HFModel *model, char *pattern);

/* Download a file from the model repo */
int hf_download_file(HFHub *h, HFModel *model, HFFile *file,
                     HttpProgressFn progress, void *arg);

/* Download file to specific path */
int hf_download_file_to(HFHub *h, HFModel *model, HFFile *file,
                        char *localpath, HttpProgressFn progress, void *arg);

/* Get cached file path (nil if not cached) */
char *hf_get_cached_path(HFHub *h, char *repo_id, char *filename);

/* Free model info */
void hf_model_free(HFModel *model);

/* Free file list */
void hf_files_free(HFFile *files);

/* Parse repo file listing from API response */
int hf_parse_file_list(char *json, HFFile **files);

/* URL encode a string */
char *hf_url_encode(char *s);

#endif /* HFHUB_H */
