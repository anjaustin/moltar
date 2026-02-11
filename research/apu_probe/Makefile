obj-m += mdla_read.o

# Multi-object module: cfi_stub MUST be first so __cfi_check is at
# the start of .text (page-aligned, required by kernel CFI shadow).
# mdla_read_main.c is the renamed mdla_read.c to avoid name collision
# with the final mdla_read.ko output.
mdla_read-objs := cfi_stub.o mdla_read_main.o

KDIR := /home/ztflynn/001/moltar/kernel-mtk
ARCH := arm64
CROSS_COMPILE := aarch64-linux-gnu-

all:
	make -C $(KDIR) M=$(PWD) ARCH=$(ARCH) CC=clang \
		CROSS_COMPILE=$(CROSS_COMPILE) CLANG_TRIPLE=aarch64-linux-gnu- \
		KCFLAGS="-msign-return-address=all -mbranch-protection=pac-ret+leaf -ffixed-x18 -fsanitize=shadow-call-stack" \
		modules

clean:
	make -C $(KDIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) clean
