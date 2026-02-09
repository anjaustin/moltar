# L-Cache VDB — Makefile (split storage)
# A vector database that fits in L-cache

ARCH := $(shell uname -m)

ifeq ($(ARCH),aarch64)
    CC = gcc
    AS = as
    SIZE = size
    NM = nm
    RUN =
else
    CC = aarch64-linux-gnu-gcc
    AS = aarch64-linux-gnu-as
    SIZE = aarch64-linux-gnu-size
    NM = aarch64-linux-gnu-nm
    RUN = qemu-aarch64
endif

CFLAGS = -O2 -Wall -march=armv8-a
ASFLAGS = -march=armv8-a

# Assembly sources (NEON distance functions)
ASM_SRCS = distance.S
# C reference sources (split storage)
C_SRCS = init_ref.c build_ref.c search_ref.c

# Common sources for all test binaries
COMMON = $(ASM_SRCS) $(C_SRCS)

.PHONY: all clean size test recall bench

all: test_lcvdb test_recall test_bench

test_lcvdb: test_lcvdb.c $(COMMON) lcvdb.h
	$(CC) $(CFLAGS) -o $@ test_lcvdb.c $(COMMON) -static

test_recall: test_recall.c $(COMMON) lcvdb.h
	$(CC) $(CFLAGS) -o $@ test_recall.c $(COMMON) -static

test_bench: test_bench.c $(COMMON) lcvdb.h
	$(CC) $(CFLAGS) -o $@ test_bench.c $(COMMON) -static

test: test_lcvdb
	$(RUN) ./test_lcvdb

recall: test_recall
	$(RUN) ./test_recall

bench: test_bench
	$(RUN) ./test_bench

size: $(COMMON)
	@echo "=== Component sizes ==="
	$(AS) $(ASFLAGS) -o distance.o distance.S
	$(CC) $(CFLAGS) -c init_ref.c -o init_ref.o
	$(CC) $(CFLAGS) -c build_ref.c -o build_ref.o
	$(CC) $(CFLAGS) -c search_ref.c -o search_ref.o
	$(SIZE) distance.o init_ref.o build_ref.o search_ref.o
	@echo ""
	@echo "=== Function sizes in binary ==="
	$(NM) --size-sort -S test_lcvdb | grep lcvdb

clean:
	rm -f test_lcvdb test_recall test_bench *.o
