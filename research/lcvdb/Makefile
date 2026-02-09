# L-Cache VDB — Makefile
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

# Assembly sources (NEON distance + init + build)
ASM_SRCS = init.S distance.S build.S
# C sources (reference search — NEON search WIP)
C_SRCS = search_ref.c

TARGET = test_lcvdb

.PHONY: all clean size test

all: $(TARGET)

$(TARGET): test_lcvdb.c $(ASM_SRCS) $(C_SRCS) lcvdb.h
	$(CC) $(CFLAGS) -o $@ test_lcvdb.c $(ASM_SRCS) $(C_SRCS) -static

test: $(TARGET)
	$(RUN) ./$(TARGET)

size: $(ASM_SRCS) $(C_SRCS)
	@echo "=== Component sizes ==="
	$(AS) $(ASFLAGS) -o init.o init.S
	$(AS) $(ASFLAGS) -o distance.o distance.S
	$(AS) $(ASFLAGS) -o build.o build.S
	$(CC) $(CFLAGS) -c search_ref.c -o search_ref.o
	$(SIZE) init.o distance.o search_ref.o build.o
	@echo ""
	@echo "=== Function sizes in binary ==="
	$(NM) --size-sort -S $(TARGET) | grep lcvdb

clean:
	rm -f $(TARGET) *.o test_debug test_debug_g test_ref test_full_ref
