CC = aarch64-linux-gnu-gcc
CFLAGS = -O2 -march=armv8.2-a+dotprod -Wall -Wextra -static
ASFLAGS = -march=armv8.2-a+dotprod

SRCS_C = colbert.c
SRCS_S = maxsim_neon.S

all: test_colbert moltar_rag

test_colbert: test_colbert.c $(SRCS_C) $(SRCS_S) colbert.h
	$(CC) $(CFLAGS) -o $@ test_colbert.c $(SRCS_C) $(SRCS_S) -lm

moltar_rag: moltar_rag.c $(SRCS_C) $(SRCS_S) colbert.h
	$(CC) $(CFLAGS) -o $@ moltar_rag.c $(SRCS_C) $(SRCS_S) -lm

clean:
	rm -f test_colbert moltar_rag

.PHONY: all clean
