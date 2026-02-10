/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Moltar MDLA 2.x Driver - Implementation
 *
 * Direct hardware access to MediaTek Deep Learning Accelerator
 * on MT6855/Dimensity 930 (MDLA 2.0/2.1)
 *
 * Uses /dev/mem for direct physical memory access (requires root)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "mdla.h"

/* Bit manipulation helpers for userspace */
#define BIT(n) (1U << (n))

#define PAGE_SIZE 4096
#define PAGE_ALIGN(x) (((x) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1))

static void *map_phys(off_t phys, size_t size) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open(/dev/mem)");
        return NULL;
    }

    off_t page_phys = phys & ~(PAGE_SIZE - 1);
    off_t offset = phys - page_phys;
    size_t page_size = PAGE_ALIGN(size + offset);

    void *base = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, page_phys);
    close(fd);

    if (base == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    return (char *)base + offset;
}

static void unmap_phys(void *addr, size_t size) {
    if (addr && addr != MAP_FAILED) {
        munmap(addr, PAGE_ALIGN(size));
    }
}

static uint32_t read32(void *base, off_t offset) {
    return *(volatile uint32_t *)((char *)base + offset);
}

static void write32(void *base, off_t offset, uint32_t val) {
    *(volatile uint32_t *)((char *)base + offset) = val;
}

int mdla_open(mdla_t *dev) {
    memset(dev, 0, sizeof(*dev));

    dev->mdla_base = map_phys(MDLA_BASE, MDLA_SIZE);
    if (!dev->mdla_base) {
        fprintf(stderr, "Failed to map MDLA registers\n");
        return -1;
    }

    dev->spm_base = map_phys(SPM_BASE, SPM_SIZE);
    if (!dev->spm_base) {
        fprintf(stderr, "Failed to map SPM registers\n");
        unmap_phys(dev->mdla_base, MDLA_SIZE);
        return -1;
    }

    dev->rpc_base = map_phys(APU_RPC_BASE, APU_RPC_SIZE);
    if (!dev->rpc_base) {
        fprintf(stderr, "Failed to map APU_RPC registers\n");
        unmap_phys(dev->spm_base, SPM_SIZE);
        unmap_phys(dev->mdla_base, MDLA_SIZE);
        return -1;
    }

    dev->iommu_base = map_phys(APU_IOMMU_BASE, APU_IOMMU_SIZE);
    if (!dev->iommu_base) {
        fprintf(stderr, "Failed to map APU_IOMMU registers\n");
        unmap_phys(dev->rpc_base, APU_RPC_SIZE);
        unmap_phys(dev->spm_base, SPM_SIZE);
        unmap_phys(dev->mdla_base, MDLA_SIZE);
        return -1;
    }

    printf("[mdla] Mapped registers:\n");
    printf("[mdla]   MDLA:  %p\n", dev->mdla_base);
    printf("[mdla]   SPM:   %p\n", dev->spm_base);
    printf("[mdla]   RPC:   %p\n", dev->rpc_base);
    printf("[mdla]   IOMMU: %p\n", dev->iommu_base);

    return 0;
}

void mdla_close(mdla_t *dev) {
    if (dev->iommu_base) unmap_phys(dev->iommu_base, APU_IOMMU_SIZE);
    if (dev->rpc_base) unmap_phys(dev->rpc_base, APU_RPC_SIZE);
    if (dev->spm_base) unmap_phys(dev->spm_base, SPM_SIZE);
    if (dev->mdla_base) unmap_phys(dev->mdla_base, MDLA_SIZE);
    memset(dev, 0, sizeof(*dev));
}

int mdla_power_on(mdla_t *dev) {
    uint32_t val;

    printf("[mdla] Power on sequence...\n");

    /* TODO: Implement proper RPC power-on sequence
     * This requires writing to RPC_SW_FIFO and related registers
     * For now, just clear reset and enable clocks
     */

    /* Clear reset */
    write32(dev->mdla_base, REG_MDLA_SW_RST, MDLA_SW_RST_MASK);
    val = read32(dev->mdla_base, REG_MDLA_SW_RST);
    printf("[mdla] SW_RST after clear: 0x%08x\n", val);

    /* Enable clocks - clear clock gating */
    write32(dev->mdla_base, REG_MDLA_CG_CLR, 0xFFFFFFFF);
    val = read32(dev->mdla_base, REG_MDLA_CG_CON);
    printf("[mdla] CG_CON: 0x%08x (expect 0 for enabled)\n", val);

    /* Check power status in SPM */
    val = read32(dev->spm_base, SPM_OTHER_PWR_STATUS);
    printf("[mdla] SPM_OTHER_PWR_STATUS: 0x%08x\n", val);
    printf("[mdla] APU power bit (5): %s\n",
           (val & APU_PWR_BIT) ? "ON" : "OFF");

    return 0;
}

int mdla_power_off(mdla_t *dev) {
    printf("[mdla] Power off sequence...\n");

    /* Assert reset */
    write32(dev->mdla_base, REG_MDLA_SW_RST, 0);

    /* Enable clock gating */
    write32(dev->mdla_base, REG_MDLA_CG_SET, 0xFFFFFFFF);

    return 0;
}

bool mdla_is_powered(mdla_t *dev) {
    uint32_t val = read32(dev->spm_base, SPM_OTHER_PWR_STATUS);
    return !!(val & APU_PWR_BIT);
}

int mdla_reset(mdla_t *dev) {
    printf("[mdla] Resetting MDLA...\n");

    /* Assert reset */
    write32(dev->mdla_base, REG_MDLA_SW_RST, MDLA_SW_RST_MASK);
    usleep(100);

    /* Clear reset */
    write32(dev->mdla_base, REG_MDLA_SW_RST, 0);
    usleep(100);

    uint32_t val = read32(dev->mdla_base, REG_MDLA_SW_RST);
    printf("[mdla] SW_RST: 0x%08x\n", val);

    return 0;
}

uint32_t mdla_read_status(mdla_t *dev) {
    return read32(dev->mdla_base, REG_MREG_TOP_G_ENG0);
}

bool mdla_is_idle(mdla_t *dev) {
    uint32_t idle = read32(dev->mdla_base, REG_MREG_TOP_G_IDLE);
    return !!(idle & 0x1);
}

uint32_t mdla_read_finish(mdla_t *dev) {
    return read32(dev->mdla_base, REG_MREG_TOP_G_FIN0);
}

int mdla_write_cmd(mdla_t *dev, uint32_t offset, uint32_t value) {
    write32(dev->mdla_base, offset, value);
    return 0;
}

uint32_t mdla_read_cmd(mdla_t *dev, uint32_t offset) {
    return read32(dev->mdla_base, offset);
}

uint32_t mdla_read_intr(mdla_t *dev) {
    return read32(dev->mdla_base, REG_MREG_TOP_G_INTP0);
}

void mdla_clear_intr(mdla_t *dev, uint32_t mask) {
    write32(dev->mdla_base, REG_MREG_TOP_G_INTP0, mask);
}

void mdla_dump_regs(mdla_t *dev) {
    printf("\n=== MDLA Register Dump ===\n");

    printf("\n--- Control Registers ---\n");
    printf("MDLA_CG_CON:   0x%08x\n", read32(dev->mdla_base, REG_MDLA_CG_CON));
    printf("MDLA_CG_SET:   0x%08x\n", read32(dev->mdla_base, REG_MDLA_CG_SET));
    printf("MDLA_CG_CLR:   0x%08x\n", read32(dev->mdla_base, REG_MDLA_CG_CLR));
    printf("MDLA_SW_RST:   0x%08x\n", read32(dev->mdla_base, REG_MDLA_SW_RST));
    printf("MDLA_CTRL:     0x%08x\n", read32(dev->mdla_base, REG_MDLA_CTRL));
    printf("MDLA_CSYSREQ:  0x%08x\n", read32(dev->mdla_base, REG_MDLA_CSYSREQ));
    printf("MDLA_CSYSACK:  0x%08x\n", read32(dev->mdla_base, REG_MDLA_CSYSACK));

    printf("\n--- Command Interface ---\n");
    printf("MREG_TOP_G_INTP0:  0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_INTP0));
    printf("MREG_TOP_G_CDMA0:  0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_CDMA0));
    printf("MREG_TOP_G_CDMA1:  0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_CDMA1));
    printf("MREG_TOP_G_CDMA2:  0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_CDMA2));
    printf("MREG_TOP_G_CDMA3:  0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_CDMA3));
    printf("MREG_TOP_G_FIN0:   0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_FIN0));
    printf("MREG_TOP_G_FIN1:   0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_FIN1));
    printf("MREG_TOP_G_IDLE:   0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_IDLE));
    printf("MREG_TOP_G_ENG0:   0x%08x\n", read32(dev->mdla_base, REG_MREG_TOP_G_ENG0));

    printf("\n--- Power Status ---\n");
    printf("SPM_OTHER_PWR_STATUS: 0x%08x\n", read32(dev->spm_base, SPM_OTHER_PWR_STATUS));

    printf("\n");
}
