# Linux Makefile for 9ml

CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

.PHONY: all test clean

all: export test/harness

export: src/export.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test/harness: test/*.c test/*.h
	$(CC) $(CFLAGS) -o $@ test/*.c $(LDFLAGS)

test: test/harness export stories15M_q80.bin
	cd test && ./harness

stories15M_q80.bin: export stories15M.bin
	./export quantize stories15M.bin stories15M_q80.bin

clean:
	rm -f export test/harness stories15M_q80.bin
