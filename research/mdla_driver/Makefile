# Moltar MDLA Driver Makefile
#
# Builds userspace MDLA driver for MT6855/Dimensity 930

CC = aarch64-linux-gnu-gcc
CFLAGS = -O2 -Wall -Wextra -g
LDFLAGS =

TARGETS = test_mdla

.PHONY: all clean test

all: $(TARGETS)

test_mdla: test_mdla.c mdla.c mdla.h
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGETS)

# Build for Android/Termux (native ARM64)
android: test_mdla.c mdla.c mdla.h
	$(CC) $(CFLAGS) -o test_mdla_native $^ $(LDFLAGS)

# Build for host x86_64 (simulation only - won't access real hardware)
host: test_mdla.c mdla.c mdla.h
	gcc $(CFLAGS) -o test_mdla_host $^ $(LDFLAGS)

help:
	@echo "Targets:"
	@echo "  all     - Build test_mdla (cross-compile for ARM64)"
	@echo "  android - Build native ARM64 binary for Termux"
	@echo "  host    - Build x86_64 for testing (simulation only)"
	@echo "  clean   - Remove built files"
	@echo ""
	@echo "Usage:"
	@echo "  1. Cross-compile for device:"
	@echo "     make"
	@echo ""
	@echo "  2. Push to device and run:"
	@echo "     adb push test_mdla /data/local/tmp/"
	@echo "     adb shell 'su -c /data/local/tmp/test_mdla'"
	@echo ""
	@echo "  3. Build native on device (Termux):"
	@echo "     make android"
	@echo ""
	@echo "Requirements:"
	@echo "  - Root access on device"
	@echo "  - /dev/mem access (may need: su -c 'chmod 666 /dev/mem')"
