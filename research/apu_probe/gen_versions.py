#!/usr/bin/env python3
"""Generate a raw __versions section binary file."""

import struct
import sys

SYMBOL_CRCS = {
    "module_layout": 0x7C24B32D,
    "__ioremap": 0x6B4B2933,
    "iounmap": 0xEDC03953,
    "printk": 0xC5850110,
    "proc_create": 0x20FD21C6,
    "remove_proc_entry": 0x3C651057,
    "arm64_use_ng_mappings": 0xAF56600A,
    "__log_post_read_mmio": 0x6980EA4B,
    "__log_read_mmio": 0xCF1211A8,
    "seq_lseek": 0xAD9F2705,
    "seq_read": 0xB9997D36,
    "seq_printf": 0xD740362B,
    "single_open": 0x4F731E2B,
    "single_release": 0xEF42EDDB,
    "__tracepoint_rwmmio_post_read": 0x19EBF04E,
    "__tracepoint_rwmmio_read": 0xA035D76E,
    "__stack_chk_fail": 0x98A9D10C,
    "__stack_chk_guard": 0x8F678B07,
    "kfree": 0x037A0CBA,
    "kmalloc_caches": 0x8900B200,
    "kmem_cache_alloc_trace": 0xB38391E9,
    "vsnprintf": 0x00148653,
    "vmalloc": 0xD6EE688F,
    "vfree": 0x999E8297,
    "__tracepoint_rwmmio_write": 0x95575C33,
    "__log_write_mmio": 0x31DFD5CD,
    "__const_udelay": 0xEAE3DFD6,
    "__arm_smccc_smc": 0xF93AAE46,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.bin> [symbol1 symbol2 ...]")
        sys.exit(1)

    output = sys.argv[1]
    if len(sys.argv) > 2:
        symbols = sys.argv[2:]
    else:
        # Default: all known symbols, module_layout first
        symbols = ["module_layout"] + [s for s in SYMBOL_CRCS if s != "module_layout"]

    data = b""
    for sym in symbols:
        crc = SYMBOL_CRCS.get(sym, 0)
        if crc == 0 and sym in SYMBOL_CRCS:
            crc = SYMBOL_CRCS[sym]
        elif crc == 0:
            print(f"WARNING: No CRC for {sym}")

        # Each entry: 4 bytes CRC + 4 bytes pad + 56 bytes name = 64 bytes
        entry = struct.pack("<I", crc)
        entry += b"\x00" * 4
        name = sym.encode("ascii") + b"\x00"
        name = name.ljust(56, b"\x00")
        entry += name
        assert len(entry) == 64, f"Entry size {len(entry)} != 64"
        data += entry

    with open(output, "wb") as f:
        f.write(data)

    print(f"Generated {output}: {len(symbols)} entries, {len(data)} bytes")


if __name__ == "__main__":
    main()
