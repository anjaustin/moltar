/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Moltar MDLA Driver Test Program
 *
 * Tests basic MDLA hardware access on MT6855/Dimensity 930
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "mdla.h"

/* Bit manipulation helpers */
#define BIT(n) (1U << (n))

int main(int argc, char *argv[]) {
    mdla_t dev;
    uint32_t status;
    int ret = 0;

    printf("=== Moltar MDLA Driver Test ===\n\n");

    if (mdla_open(&dev) < 0) {
        fprintf(stderr, "Failed to open MDLA device\n");
        return 1;
    }

    /* Dump initial register state */
    mdla_dump_regs(&dev);

    /* Check power status */
    printf("\n--- Power Status ---\n");
    if (mdla_is_powered(&dev)) {
        printf("APU power: ON\n");
    } else {
        printf("APU power: OFF\n");
        printf("WARNING: APU may be disabled in hardware!\n");
    }

    /* Try to power on */
    printf("\n--- Power On Sequence ---\n");
    mdla_power_on(&dev);

    /* Dump registers after power on */
    mdla_dump_regs(&dev);

    /* Reset the MDLA */
    printf("\n--- Reset Test ---\n");
    mdla_reset(&dev);

    /* Check status */
    printf("\n--- Status Check ---\n");
    status = mdla_read_status(&dev);
    printf("ENG status: 0x%08x\n", status);

    printf("\nIDLE status: %s\n", mdla_is_idle(&dev) ? "IDLE" : "BUSY");

    /* Check interrupt status */
    printf("\n--- Interrupt Status ---\n");
    uint32_t intr = mdla_read_intr(&dev);
    printf("INTR: 0x%08x\n", intr);

    if (intr & INTR_SWCMD_DONE) {
        printf("  - SWCMD_DONE interrupt pending\n");
    }
    if (intr & INTR_SWCMD_TILECNT_INT) {
        printf("  - TILECNT interrupt pending\n");
    }

    /* Clear any pending interrupts */
    mdla_clear_intr(&dev, intr);

    printf("\n--- Completion ---\n");
    uint32_t fin = mdla_read_finish(&dev);
    printf("FIN0: 0x%08x\n", fin);

    /* Cleanup */
    printf("\n--- Cleanup ---\n");
    mdla_power_off(&dev);
    mdla_close(&dev);

    printf("\n=== Test Complete ===\n");
    printf("Next steps: Implement command submission via CDMA registers\n");

    return ret;
}
