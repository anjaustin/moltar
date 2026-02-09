# Moltar Memory — Makefile
# Three-layer memory integration: projection + LCVDB + llama.cpp

ARCH := $(shell uname -m)

ifeq ($(ARCH),aarch64)
    CC = gcc
else
    CC = aarch64-linux-gnu-gcc
endif

CFLAGS = -O2 -Wall -march=armv8-a

# NDK compiler for dynamic-linked binaries (linking against libllama.so)
NDK = /opt/android-ndk-r27c
NDK_CC = $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang
NDK_CFLAGS = -O2 -Wall -march=armv8.2-a+dotprod+fp16

# LCVDB sources (from ../lcvdb/)
LCVDB_DIR = ../lcvdb
LCVDB_SRCS = $(LCVDB_DIR)/distance.S $(LCVDB_DIR)/init_ref.c \
             $(LCVDB_DIR)/build_ref.c $(LCVDB_DIR)/search_ref.c
LCVDB_INC = -I$(LCVDB_DIR)

# llama.cpp headers and libraries
LLAMA_DIR = ../llama.cpp
LLAMA_INC = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LLAMA_LIB_DIR = $(LLAMA_DIR)/build-android/bin
LLAMA_LIBS = -L$(LLAMA_LIB_DIR) -lllama -lggml -lggml-base -lggml-cpu

# Memory layer sources
MEM_SRCS = project.c

.PHONY: all agent server clean test

all: test_project

agent: moltar-agent

server: moltar-server

# Static test binary (cross-compiled with GNU toolchain)
test_project: test_project.c $(MEM_SRCS) $(LCVDB_SRCS)
	$(CC) $(CFLAGS) $(LCVDB_INC) -o $@ $^ -static -lm

# Dynamic agent binary (NDK, links against libllama.so on device)
moltar-agent: moltar-agent.c $(MEM_SRCS) $(LCVDB_SRCS)
	$(NDK_CC) $(NDK_CFLAGS) $(LCVDB_INC) $(LLAMA_INC) -o $@ $^ \
		$(LLAMA_LIBS) -lm -llog -landroid \
		-Wl,-rpath,/data/local/tmp -Wl,-rpath,.

# Dynamic server binary (NDK, links against libllama.so on device)
moltar-server: moltar-server.c $(MEM_SRCS) $(LCVDB_SRCS)
	$(NDK_CC) $(NDK_CFLAGS) $(LCVDB_INC) $(LLAMA_INC) -o $@ $^ \
		$(LLAMA_LIBS) -lm -llog -landroid \
		-Wl,-rpath,/data/local/tmp -Wl,-rpath,.

test: test_project
	./test_project

clean:
	rm -f test_project moltar-agent moltar-server *.o
