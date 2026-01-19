/*
 * llmfs - 9P file server for LLM inference
 * Plan 9 port of llama2.c with remote inference support
 *
 * File structure:
 *   /ctl        - server control (load, unload, pool-*, server status)
 *   /model      - model info (read-only)
 *   /pool/      - pool info directory
 *     count     - number of loaded models
 *     memory    - total memory used
 *     list      - list of loaded model names
 *   /clone      - read to create new session, returns session id
 *   /N/         - session N directory
 *     ctl       - session control (temp, topp, seed, steps, generate, reset, model)
 *     model     - current model name (read/write)
 *     prompt    - write prompt text here
 *     output    - read complete output (blocks until done)
 *     stream    - read streaming output (returns tokens as generated)
 *     status    - current status (idle, generating N/M, done tok/s, error)
 *
 * Server /ctl commands:
 *   load <model> <tokenizer>           - load model (legacy, single-model)
 *   unload                             - unload model (legacy)
 *   pool-load <name> <model> <tok>     - load model into pool
 *   pool-unload <name>                 - unload model from pool
 *   pool-limit <max_models> <max_mem>  - set pool limits
 */

/* Include model implementation first (it includes u.h, libc.h, and thread.h) */
#include "model.c"

/* Include pool implementation */
#include "pool/pool.c"

/* Additional headers for 9P server */
#include <fcall.h>
#include <9p.h>

/* Maximum sessions */
#define MAX_SESSIONS 16

/* Default pool limits */
#define DEFAULT_MAX_MODELS 8
#define DEFAULT_MAX_MEMORY (4ULL * 1024 * 1024 * 1024)  /* 4GB */

/* Session states */
enum {
	SESS_IDLE,
	SESS_GENERATING,
	SESS_DONE,
	SESS_ERROR
};

/* File types */
enum {
	Qroot,
	Qctl,
	Qmodel,
	Qclone,
	Qpool,          /* pool/ directory */
	Qpoolcount,     /* pool/count */
	Qpoolmemory,    /* pool/memory */
	Qpoollist,      /* pool/list */
	Qsession,
	Qsessctl,
	Qsessmodel,     /* session model binding */
	Qprompt,
	Qoutput,
	Qstream,
	Qstatus
};

/* Session structure */
typedef struct Session Session;
struct Session {
	int id;
	int state;
	int inuse;

	/* Model binding */
	PoolEntry *model;       /* bound model from pool (nil = use default) */
	char modelname[64];     /* name of bound model */

	/* Generation parameters */
	float temp;
	float topp;
	uvlong seed;
	int steps;

	/* Prompt */
	char *prompt;
	int promptlen;
	int promptcap;

	/* Output */
	char *output;
	int outlen;
	int outcap;

	/* Streaming */
	Channel *tokenchan;     /* buffered channel for streaming tokens */
	char *streambuf;        /* accumulated stream data */
	int streamlen;
	int streamcap;
	int streampos;          /* read position in streambuf */

	/* Stats */
	int genpos;             /* tokens generated so far */
	vlong startns;
	double toksec;
	char errmsg[256];

	/* Locks */
	QLock lk;
};

/* Server state */
typedef struct LLMServer LLMServer;
struct LLMServer {
	int loaded;                     /* Backward compat: 1 if default model loaded */
	char *modelpath;                /* Default model path */
	char *tokenizerpath;            /* Default tokenizer path */
	Transformer transformer;        /* Default model (for backward compat) */
	Tokenizer tokenizer;            /* Default tokenizer */
	ModelPool pool;                 /* Model pool for multi-model support */
	PoolEntry *defaultmodel;        /* Default model entry in pool */
	Session sessions[MAX_SESSIONS];
	int nsessions;
	QLock lk;
};

static LLMServer server;
static char srvname[64];

/* Forward declarations */
static void genproc(void *arg);
static void fsread(Req *r);
static void fswrite(Req *r);
static void fsopen(Req *r);
static void fsstat(Req *r);
static char *fswalk1(Fid *fid, char *name, Qid *qid);
static char *fsclone(Fid *oldfid, Fid *newfid);
static void fsattach(Req *r);
static void fsdestroyfid(Fid *fid);

/* QID path encoding: high 8 bits = type, low 24 bits = session id */
#define QPATH(type, sessid) (((uvlong)(type) << 24) | ((sessid) & 0xFFFFFF))
#define QTYPE(path) ((int)((path) >> 24))
#define QSESS(path) ((int)((path) & 0xFFFFFF))

/* Directory entry template */
static void
mkqid(Qid *qid, int type, int sessid)
{
	qid->path = QPATH(type, sessid);
	qid->vers = 0;
	qid->type = (type == Qroot || type == Qsession || type == Qpool) ? QTDIR : QTFILE;
}

static void
mkdir(Dir *d, int type, int sessid, char *name)
{
	memset(d, 0, sizeof(*d));
	mkqid(&d->qid, type, sessid);
	d->mode = (d->qid.type == QTDIR) ? (DMDIR | 0755) : 0644;
	d->atime = d->mtime = time(0);
	d->length = 0;
	d->name = estrdup9p(name);
	d->uid = estrdup9p("llm");
	d->gid = estrdup9p("llm");
	d->muid = estrdup9p("llm");

	/* Make prompt write-only, output/stream/model/status read-only */
	if (type == Qprompt)
		d->mode = 0222;
	else if (type == Qoutput || type == Qstream || type == Qmodel || type == Qstatus)
		d->mode = 0444;
	else if (type == Qpoolcount || type == Qpoolmemory || type == Qpoollist)
		d->mode = 0444;  /* Pool info is read-only */
	else if (type == Qctl || type == Qsessctl || type == Qclone || type == Qsessmodel)
		d->mode = 0666;  /* Session model binding is read-write */
}

/* Session management */
static Session*
session_new(void)
{
	Session *s;
	int i;

	qlock(&server.lk);
	for (i = 0; i < MAX_SESSIONS; i++) {
		s = &server.sessions[i];
		if (!s->inuse) {
			memset(s, 0, sizeof(*s));
			s->id = i;
			s->inuse = 1;
			s->state = SESS_IDLE;
			s->temp = 1.0f;
			s->topp = 0.9f;
			s->seed = nsec();
			s->steps = 256;
			s->promptcap = 1024;
			s->prompt = malloc(s->promptcap);
			s->outcap = 4096;
			s->output = malloc(s->outcap);
			s->streamcap = 4096;
			s->streambuf = malloc(s->streamcap);
			s->tokenchan = chancreate(sizeof(char*), 64);
			if (i >= server.nsessions)
				server.nsessions = i + 1;
			qunlock(&server.lk);
			return s;
		}
	}
	qunlock(&server.lk);
	return nil;
}

static Session*
session_get(int id)
{
	if (id < 0 || id >= MAX_SESSIONS)
		return nil;
	if (!server.sessions[id].inuse)
		return nil;
	return &server.sessions[id];
}

static void
session_free(Session *s)
{
	if (s == nil)
		return;

	qlock(&s->lk);
	/* Release model binding */
	if (s->model != nil) {
		pool_release(&server.pool, s->model);
		s->model = nil;
		s->modelname[0] = '\0';
	}
	s->inuse = 0;
	free(s->prompt);
	s->prompt = nil;
	free(s->output);
	s->output = nil;
	free(s->streambuf);
	s->streambuf = nil;
	if (s->tokenchan) {
		/* Drain channel */
		char *tok;
		while (nbrecv(s->tokenchan, &tok) > 0) {
			if (tok)
				free(tok);
		}
		chanfree(s->tokenchan);
		s->tokenchan = nil;
	}
	qunlock(&s->lk);
}

static void
session_reset(Session *s)
{
	qlock(&s->lk);
	s->state = SESS_IDLE;
	s->promptlen = 0;
	s->outlen = 0;
	s->streamlen = 0;
	s->streampos = 0;
	s->genpos = 0;
	s->toksec = 0;
	s->errmsg[0] = '\0';

	/* Drain token channel */
	char *tok;
	while (nbrecv(s->tokenchan, &tok) > 0) {
		if (tok)
			free(tok);
	}
	qunlock(&s->lk);
}

static void
session_append_output(Session *s, char *piece)
{
	int len = strlen(piece);

	if (s->outlen + len + 1 > s->outcap) {
		s->outcap *= 2;
		s->output = realloc(s->output, s->outcap);
	}
	memcpy(s->output + s->outlen, piece, len);
	s->outlen += len;
	s->output[s->outlen] = '\0';

	if (s->streamlen + len + 1 > s->streamcap) {
		s->streamcap *= 2;
		s->streambuf = realloc(s->streambuf, s->streamcap);
	}
	memcpy(s->streambuf + s->streamlen, piece, len);
	s->streamlen += len;
	s->streambuf[s->streamlen] = '\0';
}

/* Generation thread */
static void
genproc(void *arg)
{
	Session *s = arg;
	Sampler sampler;
	int *prompt_tokens;
	int num_prompt_tokens;
	int token, next, pos;
	char *piece;
	float *logits;
	vlong startns;
	Transformer *transformer;
	Tokenizer *tokenizer;

	/* Get model to use: session-bound model or default */
	if (s->model != nil) {
		transformer = (Transformer *)s->model->transformer;
		tokenizer = (Tokenizer *)s->model->tokenizer;
	} else if (server.loaded) {
		transformer = &server.transformer;
		tokenizer = &server.tokenizer;
	} else {
		qlock(&s->lk);
		s->state = SESS_ERROR;
		strcpy(s->errmsg, "no model loaded");
		qunlock(&s->lk);
		sendp(s->tokenchan, nil);
		return;
	}

	/* Encode prompt */
	char *prompt = s->prompt;
	if (prompt == nil || prompt[0] == '\0')
		prompt = "";

	prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
	encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

	if (num_prompt_tokens < 1) {
		qlock(&s->lk);
		s->state = SESS_ERROR;
		strcpy(s->errmsg, "encoding failed");
		qunlock(&s->lk);
		free(prompt_tokens);
		sendp(s->tokenchan, nil);
		return;
	}

	/* Build sampler */
	build_sampler(&sampler, transformer->config.vocab_size, s->temp, s->topp, s->seed);

	/* Determine steps */
	int steps = s->steps;
	if (steps <= 0 || steps > transformer->config.seq_len)
		steps = transformer->config.seq_len;

	/* Start generating */
	token = prompt_tokens[0];
	startns = nsec();
	s->startns = startns;
	pos = 0;

	while (pos < steps) {
		logits = forward(transformer, token, pos);

		if (pos < num_prompt_tokens - 1) {
			next = prompt_tokens[pos + 1];
		} else {
			next = sample(&sampler, logits);
		}
		pos++;

		if (next == 1) /* EOS */
			break;

		piece = decode(tokenizer, token, next);

		/* Send token to channel */
		char *dup = estrdup9p(piece);
		if (sendp(s->tokenchan, dup) < 0) {
			free(dup);
			break;
		}

		/* Also append to output buffer */
		qlock(&s->lk);
		session_append_output(s, piece);
		s->genpos = pos;
		qunlock(&s->lk);

		token = next;
	}

	/* Signal EOF */
	sendp(s->tokenchan, nil);

	/* Update stats */
	qlock(&s->lk);
	vlong endns = nsec();
	if (pos > 1)
		s->toksec = (double)(pos - 1) / ((endns - startns) / 1000000000.0);
	s->state = SESS_DONE;
	qunlock(&s->lk);

	free_sampler(&sampler);
	free(prompt_tokens);
}

/* Server operations */
static void
loadmodel(char *modelpath, char *tokpath)
{
	qlock(&server.lk);

	if (server.loaded) {
		free_tokenizer(&server.tokenizer);
		free_transformer(&server.transformer);
		free(server.modelpath);
		free(server.tokenizerpath);
		server.loaded = 0;
	}

	build_transformer(&server.transformer, modelpath);
	build_tokenizer(&server.tokenizer, tokpath, server.transformer.config.vocab_size);

	server.modelpath = estrdup9p(modelpath);
	server.tokenizerpath = estrdup9p(tokpath);
	server.loaded = 1;

	qunlock(&server.lk);
}

static void
unloadmodel(void)
{
	qlock(&server.lk);
	if (server.loaded) {
		free_tokenizer(&server.tokenizer);
		free_transformer(&server.transformer);
		free(server.modelpath);
		free(server.tokenizerpath);
		server.modelpath = nil;
		server.tokenizerpath = nil;
		server.loaded = 0;
	}
	qunlock(&server.lk);
}

/* 9P operations */
static void
fsattach(Req *r)
{
	r->fid->qid.path = QPATH(Qroot, 0);
	r->fid->qid.type = QTDIR;
	r->fid->qid.vers = 0;
	r->ofcall.qid = r->fid->qid;
	respond(r, nil);
}

static char*
fswalk1(Fid *fid, char *name, Qid *qid)
{
	int type = QTYPE(fid->qid.path);
	int sessid = QSESS(fid->qid.path);
	int i;

	if (type == Qroot) {
		if (strcmp(name, "..") == 0) {
			mkqid(qid, Qroot, 0);
			return nil;
		}
		if (strcmp(name, "ctl") == 0) {
			mkqid(qid, Qctl, 0);
			return nil;
		}
		if (strcmp(name, "model") == 0) {
			mkqid(qid, Qmodel, 0);
			return nil;
		}
		if (strcmp(name, "clone") == 0) {
			mkqid(qid, Qclone, 0);
			return nil;
		}
		if (strcmp(name, "pool") == 0) {
			mkqid(qid, Qpool, 0);
			return nil;
		}
		/* Check for session directory */
		i = atoi(name);
		if (i >= 0 && i < server.nsessions && server.sessions[i].inuse) {
			mkqid(qid, Qsession, i);
			return nil;
		}
		return "file not found";
	}

	if (type == Qpool) {
		if (strcmp(name, "..") == 0) {
			mkqid(qid, Qroot, 0);
			return nil;
		}
		if (strcmp(name, "count") == 0) {
			mkqid(qid, Qpoolcount, 0);
			return nil;
		}
		if (strcmp(name, "memory") == 0) {
			mkqid(qid, Qpoolmemory, 0);
			return nil;
		}
		if (strcmp(name, "list") == 0) {
			mkqid(qid, Qpoollist, 0);
			return nil;
		}
		return "file not found";
	}

	if (type == Qsession) {
		if (strcmp(name, "..") == 0) {
			mkqid(qid, Qroot, 0);
			return nil;
		}
		if (strcmp(name, "ctl") == 0) {
			mkqid(qid, Qsessctl, sessid);
			return nil;
		}
		if (strcmp(name, "model") == 0) {
			mkqid(qid, Qsessmodel, sessid);
			return nil;
		}
		if (strcmp(name, "prompt") == 0) {
			mkqid(qid, Qprompt, sessid);
			return nil;
		}
		if (strcmp(name, "output") == 0) {
			mkqid(qid, Qoutput, sessid);
			return nil;
		}
		if (strcmp(name, "stream") == 0) {
			mkqid(qid, Qstream, sessid);
			return nil;
		}
		if (strcmp(name, "status") == 0) {
			mkqid(qid, Qstatus, sessid);
			return nil;
		}
		return "file not found";
	}

	return "walk in non-directory";
}

static char*
fsclone(Fid *oldfid, Fid *newfid)
{
	USED(oldfid);
	USED(newfid);
	return nil;
}

static void
fsdestroyfid(Fid *fid)
{
	USED(fid);
}

static int
rootgen(int n, Dir *d, void *aux)
{
	USED(aux);

	switch (n) {
	case 0:
		mkdir(d, Qctl, 0, "ctl");
		return 0;
	case 1:
		mkdir(d, Qmodel, 0, "model");
		return 0;
	case 2:
		mkdir(d, Qclone, 0, "clone");
		return 0;
	case 3:
		mkdir(d, Qpool, 0, "pool");
		return 0;
	default:
		n -= 4;
		if (n >= 0 && n < server.nsessions && server.sessions[n].inuse) {
			char name[16];
			sprint(name, "%d", n);
			mkdir(d, Qsession, n, name);
			return 0;
		}
		return -1;
	}
}

static int
sessiongen(int n, Dir *d, void *aux)
{
	int sessid = (int)(uintptr)aux;

	switch (n) {
	case 0:
		mkdir(d, Qsessctl, sessid, "ctl");
		return 0;
	case 1:
		mkdir(d, Qsessmodel, sessid, "model");
		return 0;
	case 2:
		mkdir(d, Qprompt, sessid, "prompt");
		return 0;
	case 3:
		mkdir(d, Qoutput, sessid, "output");
		return 0;
	case 4:
		mkdir(d, Qstream, sessid, "stream");
		return 0;
	case 5:
		mkdir(d, Qstatus, sessid, "status");
		return 0;
	default:
		return -1;
	}
}

static int
poolgen(int n, Dir *d, void *aux)
{
	USED(aux);

	switch (n) {
	case 0:
		mkdir(d, Qpoolcount, 0, "count");
		return 0;
	case 1:
		mkdir(d, Qpoolmemory, 0, "memory");
		return 0;
	case 2:
		mkdir(d, Qpoollist, 0, "list");
		return 0;
	default:
		return -1;
	}
}

static void
fsread(Req *r)
{
	int type = QTYPE(r->fid->qid.path);
	int sessid = QSESS(r->fid->qid.path);
	Session *s;
	char buf[4096];
	int n;
	char *tok;

	switch (type) {
	case Qroot:
		dirread9p(r, rootgen, nil);
		respond(r, nil);
		return;

	case Qsession:
		dirread9p(r, sessiongen, (void*)(uintptr)sessid);
		respond(r, nil);
		return;

	case Qctl:
		n = 0;
		if (server.loaded)
			n = snprint(buf, sizeof(buf), "loaded %s %s\n", server.modelpath, server.tokenizerpath);
		else
			n = snprint(buf, sizeof(buf), "unloaded\n");
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qmodel:
		if (!server.loaded) {
			respond(r, "model not loaded");
			return;
		}
		{
			Config *c = &server.transformer.config;
			n = snprint(buf, sizeof(buf),
				"dim %d\n"
				"hidden_dim %d\n"
				"n_layers %d\n"
				"n_heads %d\n"
				"n_kv_heads %d\n"
				"vocab_size %d\n"
				"seq_len %d\n",
				c->dim, c->hidden_dim, c->n_layers,
				c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);
		}
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qclone:
		s = session_new();
		if (s == nil) {
			respond(r, "too many sessions");
			return;
		}
		n = snprint(buf, sizeof(buf), "%d\n", s->id);
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qsessctl:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}
		qlock(&s->lk);
		n = snprint(buf, sizeof(buf),
			"temp %g\ntopp %g\nseed %llud\nsteps %d\n",
			s->temp, s->topp, s->seed, s->steps);
		qunlock(&s->lk);
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qstatus:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}
		qlock(&s->lk);
		switch (s->state) {
		case SESS_IDLE:
			n = snprint(buf, sizeof(buf), "idle\n");
			break;
		case SESS_GENERATING:
			n = snprint(buf, sizeof(buf), "generating %d/%d\n", s->genpos, s->steps);
			break;
		case SESS_DONE:
			n = snprint(buf, sizeof(buf), "done %.2f tok/s\n", s->toksec);
			break;
		case SESS_ERROR:
			n = snprint(buf, sizeof(buf), "error %s\n", s->errmsg);
			break;
		}
		qunlock(&s->lk);
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qoutput:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}
		/* Block until generation is done */
		while (s->state == SESS_GENERATING)
			sleep(100);

		qlock(&s->lk);
		if (s->state == SESS_ERROR) {
			qunlock(&s->lk);
			respond(r, s->errmsg);
			return;
		}
		readbuf(r, s->output, s->outlen);
		qunlock(&s->lk);
		respond(r, nil);
		return;

	case Qstream:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}

		/* Non-blocking receive of available tokens */
		n = 0;
		while (n < sizeof(buf) - 256) {
			if (nbrecv(s->tokenchan, &tok) <= 0)
				break;
			if (tok == nil) {
				/* EOF marker */
				break;
			}
			int len = strlen(tok);
			if (n + len >= sizeof(buf) - 1) {
				/* Put it back - can't fit */
				sendp(s->tokenchan, tok);
				break;
			}
			memcpy(buf + n, tok, len);
			n += len;
			free(tok);
		}
		buf[n] = '\0';

		if (n == 0 && s->state == SESS_DONE) {
			r->ofcall.count = 0;
			respond(r, nil);
			return;
		}

		readbuf(r, buf, n);
		respond(r, nil);
		return;

	case Qprompt:
		respond(r, "prompt is write-only");
		return;

	case Qpool:
		dirread9p(r, poolgen, nil);
		respond(r, nil);
		return;

	case Qpoolcount:
		n = snprint(buf, sizeof(buf), "%d\n", pool_count(&server.pool));
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qpoolmemory:
		n = snprint(buf, sizeof(buf), "%llud\n", pool_memory(&server.pool));
		readstr(r, buf);
		respond(r, nil);
		return;

	case Qpoollist:
		{
			char *list = pool_list(&server.pool);
			if (list == nil) {
				respond(r, "out of memory");
				return;
			}
			readstr(r, list);
			free(list);
			respond(r, nil);
			return;
		}

	case Qsessmodel:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}
		qlock(&s->lk);
		if (s->modelname[0] != '\0')
			n = snprint(buf, sizeof(buf), "%s\n", s->modelname);
		else
			n = snprint(buf, sizeof(buf), "(default)\n");
		qunlock(&s->lk);
		readstr(r, buf);
		respond(r, nil);
		return;

	default:
		respond(r, "unknown file");
		return;
	}
}

static void
fswrite(Req *r)
{
	int type = QTYPE(r->fid->qid.path);
	int sessid = QSESS(r->fid->qid.path);
	Session *s;
	char buf[1024];
	int n;

	n = r->ifcall.count;
	if (n >= sizeof(buf))
		n = sizeof(buf) - 1;
	memcpy(buf, r->ifcall.data, n);
	buf[n] = '\0';

	/* Strip trailing newline */
	while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r'))
		buf[--n] = '\0';

	switch (type) {
	case Qctl:
		if (strncmp(buf, "load ", 5) == 0) {
			char *args = buf + 5;
			char *modelpath, *tokpath;

			/* Parse: load <modelpath> <tokenizerpath> */
			modelpath = args;
			tokpath = strchr(args, ' ');
			if (tokpath == nil) {
				respond(r, "usage: load <model> <tokenizer>");
				return;
			}
			*tokpath++ = '\0';
			while (*tokpath == ' ')
				tokpath++;

			loadmodel(modelpath, tokpath);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strcmp(buf, "unload") == 0) {
			unloadmodel();
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		/* Pool commands */
		if (strncmp(buf, "pool-load ", 10) == 0) {
			char *args = buf + 10;
			char *name, *modelpath, *tokpath;

			/* Parse: pool-load <name> <modelpath> <tokenizerpath> */
			name = args;
			modelpath = strchr(args, ' ');
			if (modelpath == nil) {
				respond(r, "usage: pool-load <name> <model> <tokenizer>");
				return;
			}
			*modelpath++ = '\0';
			while (*modelpath == ' ')
				modelpath++;
			tokpath = strchr(modelpath, ' ');
			if (tokpath == nil) {
				respond(r, "usage: pool-load <name> <model> <tokenizer>");
				return;
			}
			*tokpath++ = '\0';
			while (*tokpath == ' ')
				tokpath++;

			PoolEntry *e = pool_load(&server.pool, name, modelpath, tokpath);
			if (e == nil) {
				respond(r, "failed to load model");
				return;
			}
			pool_release(&server.pool, e);  /* Don't keep a reference */
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strncmp(buf, "pool-unload ", 12) == 0) {
			char *name = buf + 12;
			while (*name == ' ')
				name++;
			if (pool_unload(&server.pool, name) != 0) {
				respond(r, "model not found or in use");
				return;
			}
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strncmp(buf, "pool-limit ", 11) == 0) {
			char *args = buf + 11;
			int max_models;
			uvlong max_memory;

			/* Parse: pool-limit <max_models> <max_memory> */
			max_models = atoi(args);
			char *memarg = strchr(args, ' ');
			if (memarg == nil) {
				respond(r, "usage: pool-limit <max_models> <max_memory_bytes>");
				return;
			}
			max_memory = strtoull(memarg + 1, nil, 10);

			qlock(&server.pool.lk);
			server.pool.max_models = max_models;
			server.pool.max_memory = max_memory;
			qunlock(&server.pool.lk);

			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		respond(r, "unknown ctl command");
		return;

	case Qsessctl:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}

		if (strncmp(buf, "temp ", 5) == 0) {
			qlock(&s->lk);
			s->temp = atof(buf + 5);
			if (s->temp < 0.0f)
				s->temp = 0.0f;
			qunlock(&s->lk);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strncmp(buf, "topp ", 5) == 0) {
			qlock(&s->lk);
			s->topp = atof(buf + 5);
			if (s->topp < 0.0f || s->topp > 1.0f)
				s->topp = 0.9f;
			qunlock(&s->lk);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strncmp(buf, "seed ", 5) == 0) {
			qlock(&s->lk);
			s->seed = strtoul(buf + 5, nil, 10);
			qunlock(&s->lk);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strncmp(buf, "steps ", 6) == 0) {
			qlock(&s->lk);
			s->steps = atoi(buf + 6);
			if (s->steps < 0)
				s->steps = 256;
			qunlock(&s->lk);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strcmp(buf, "generate") == 0) {
			qlock(&s->lk);
			if (s->state == SESS_GENERATING) {
				qunlock(&s->lk);
				respond(r, "already generating");
				return;
			}
			s->state = SESS_GENERATING;
			qunlock(&s->lk);

			/* Start generation in a new thread */
			proccreate(genproc, s, 65536);

			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strcmp(buf, "reset") == 0) {
			session_reset(s);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		if (strcmp(buf, "close") == 0) {
			session_free(s);
			r->ofcall.count = r->ifcall.count;
			respond(r, nil);
			return;
		}
		respond(r, "unknown session ctl command");
		return;

	case Qprompt:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}

		qlock(&s->lk);
		if (s->state == SESS_GENERATING) {
			qunlock(&s->lk);
			respond(r, "cannot write prompt while generating");
			return;
		}

		/* Append to prompt */
		n = r->ifcall.count;
		if (s->promptlen + n + 1 > s->promptcap) {
			while (s->promptlen + n + 1 > s->promptcap)
				s->promptcap *= 2;
			s->prompt = realloc(s->prompt, s->promptcap);
		}
		memcpy(s->prompt + s->promptlen, r->ifcall.data, n);
		s->promptlen += n;
		s->prompt[s->promptlen] = '\0';
		qunlock(&s->lk);

		r->ofcall.count = r->ifcall.count;
		respond(r, nil);
		return;

	case Qsessmodel:
		s = session_get(sessid);
		if (s == nil) {
			respond(r, "session not found");
			return;
		}
		qlock(&s->lk);
		if (s->state == SESS_GENERATING) {
			qunlock(&s->lk);
			respond(r, "cannot change model while generating");
			return;
		}
		/* Release previous model if bound */
		if (s->model != nil) {
			pool_release(&server.pool, s->model);
			s->model = nil;
			s->modelname[0] = '\0';
		}
		/* Bind to new model (if not empty/default) */
		if (buf[0] != '\0' && strcmp(buf, "(default)") != 0) {
			PoolEntry *e = pool_get(&server.pool, buf);
			if (e == nil) {
				qunlock(&s->lk);
				respond(r, "model not found in pool");
				return;
			}
			s->model = e;
			strncpy(s->modelname, buf, sizeof(s->modelname) - 1);
			s->modelname[sizeof(s->modelname) - 1] = '\0';
		}
		qunlock(&s->lk);
		r->ofcall.count = r->ifcall.count;
		respond(r, nil);
		return;

	default:
		respond(r, "cannot write to this file");
		return;
	}
}

static void
fsopen(Req *r)
{
	int type = QTYPE(r->fid->qid.path);
	int omode = r->ifcall.mode & 3;

	/* Check permissions */
	switch (type) {
	case Qprompt:
		if (omode != OWRITE) {
			respond(r, "prompt is write-only");
			return;
		}
		break;
	case Qoutput:
	case Qstream:
	case Qmodel:
	case Qstatus:
	case Qpoolcount:
	case Qpoolmemory:
	case Qpoollist:
		if (omode != OREAD) {
			respond(r, "file is read-only");
			return;
		}
		break;
	}

	respond(r, nil);
}

static void
fsstat(Req *r)
{
	int type = QTYPE(r->fid->qid.path);
	int sessid = QSESS(r->fid->qid.path);
	Dir d;
	char name[16];

	switch (type) {
	case Qroot:
		mkdir(&d, Qroot, 0, "/");
		break;
	case Qctl:
		mkdir(&d, Qctl, 0, "ctl");
		break;
	case Qmodel:
		mkdir(&d, Qmodel, 0, "model");
		break;
	case Qclone:
		mkdir(&d, Qclone, 0, "clone");
		break;
	case Qsession:
		sprint(name, "%d", sessid);
		mkdir(&d, Qsession, sessid, name);
		break;
	case Qsessctl:
		mkdir(&d, Qsessctl, sessid, "ctl");
		break;
	case Qprompt:
		mkdir(&d, Qprompt, sessid, "prompt");
		break;
	case Qoutput:
		mkdir(&d, Qoutput, sessid, "output");
		break;
	case Qstream:
		mkdir(&d, Qstream, sessid, "stream");
		break;
	case Qstatus:
		mkdir(&d, Qstatus, sessid, "status");
		break;
	case Qpool:
		mkdir(&d, Qpool, 0, "pool");
		break;
	case Qpoolcount:
		mkdir(&d, Qpoolcount, 0, "count");
		break;
	case Qpoolmemory:
		mkdir(&d, Qpoolmemory, 0, "memory");
		break;
	case Qpoollist:
		mkdir(&d, Qpoollist, 0, "list");
		break;
	case Qsessmodel:
		mkdir(&d, Qsessmodel, sessid, "model");
		break;
	default:
		respond(r, "unknown file");
		return;
	}

	r->d = d;
	r->d.name = estrdup9p(d.name);
	r->d.uid = estrdup9p(d.uid);
	r->d.gid = estrdup9p(d.gid);
	r->d.muid = estrdup9p(d.muid);
	free(d.name);
	free(d.uid);
	free(d.gid);
	free(d.muid);
	respond(r, nil);
}

Srv llmfssrv = {
	.attach = fsattach,
	.walk1 = fswalk1,
	.clone = fsclone,
	.open = fsopen,
	.read = fsread,
	.write = fswrite,
	.stat = fsstat,
	.destroyfid = fsdestroyfid,
};

static void
usage(void)
{
	fprint(2, "usage: llmfs [-s srvname] [-m mountpoint]\n");
	exits("usage");
}

void
threadmain(int argc, char *argv[])
{
	char *sname = "llm";
	char *mtpt = nil;

	ARGBEGIN {
	case 's':
		sname = EARGF(usage());
		break;
	case 'm':
		mtpt = EARGF(usage());
		break;
	default:
		usage();
	} ARGEND

	if (argc > 0)
		usage();

	/* Initialize server state */
	memset(&server, 0, sizeof(server));
	strcpy(srvname, sname);

	/* Initialize model pool */
	pool_init(&server.pool, DEFAULT_MAX_MODELS, DEFAULT_MAX_MEMORY);

	/* Post and mount the file server */
	threadpostmountsrv(&llmfssrv, sname, mtpt, MREPL|MCREATE);

	threadexits(nil);
}
