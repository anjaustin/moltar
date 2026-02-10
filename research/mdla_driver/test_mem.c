/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Test if we can access /dev/mem or create it
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <errno.h>
#include <stdint.h>

#define PAGE_SIZE 4096

int main() {
    int fd;
    void *mapped;
    
    printf("=== /dev/mem Access Test ===\n\n");
    
    /* Try to open existing /dev/mem */
    printf("Trying to open /dev/mem...\n");
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    
    if (fd < 0) {
        printf("open() failed: %s (%d)\n", strerror(errno), errno);
        
        /* Try to create it */
        printf("\nTrying to create /dev/mem with mknod...\n");
        if (mknod("/dev/mem", S_IFCHR | 0666, makedev(1, 1)) < 0) {
            printf("mknod() failed: %s (%d)\n", strerror(errno), errno);
            return 1;
        }
        printf("mknod() succeeded!\n");
        
        /* Try to open again */
        fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            printf("open() after mknod failed: %s (%d)\n", strerror(errno), errno);
            return 1;
        }
    }
    
    printf("Successfully opened /dev/mem (fd=%d)\n", fd);
    
    /* Try to map some physical addresses */
    printf("\nTrying to map SPM (0x10006000)...\n");
    mapped = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x10006000);
    
    if (mapped == MAP_FAILED) {
        printf("mmap(0x10006000) failed: %s (%d)\n", strerror(errno), errno);
    } else {
        printf("Mapped SPM at %p\n", mapped);
        
        /* Read SPM_OTHER_PWR_STATUS (offset 0x178) */
        volatile uint32_t *spm_reg = (uint32_t *)((char *)mapped + 0x178);
        uint32_t val = *spm_reg;
        printf("SPM_OTHER_PWR_STATUS: 0x%08x\n", val);
        printf("APU power bit (5): %s\n", (val & (1 << 5)) ? "ON" : "OFF");
        
        munmap(mapped, PAGE_SIZE);
    }
    
    printf("\nTrying to map MDLA (0x19034000)...\n");
    mapped = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x19034000);
    
    if (mapped == MAP_FAILED) {
        printf("mmap(0x19034000) failed: %s (%d)\n", strerror(errno), errno);
    } else {
        printf("Mapped MDLA at %p\n", mapped);
        
        /* Read MDLA_CG_CON (offset 0x000) */
        volatile uint32_t *mdla_reg = (uint32_t *)mapped;
        uint32_t val = *mdla_reg;
        printf("MDLA_CG_CON: 0x%08x\n", val);
        
        /* Read MDLA_SW_RST (offset 0x00C) */
        mdla_reg = (uint32_t *)((char *)mapped + 0x00C);
        val = *mdla_reg;
        printf("MDLA_SW_RST: 0x%08x\n", val);
        
        munmap(mapped, PAGE_SIZE);
    }
    
    close(fd);
    printf("\n=== Test Complete ===\n");
    
    return 0;
}
