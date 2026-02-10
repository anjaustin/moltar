// SPDX-License-Identifier: GPL-2.0
/*
 * Moltar APU Hardware Probe Module
 *
 * Minimal kernel module to probe MediaTek APU (MDLA 2.1) hardware
 * on the Dimensity 930 (MT6855) SoC.
 *
 * This module bypasses the device tree (since Motorola removed all APU
 * DT nodes) and directly maps physical register addresses to check if
 * the APU hardware is accessible and responsive.
 *
 * Register addresses are based on the MT6853/MT6855 kernel source
 * from the Vivo Y77 kernel tree.
 *
 * Usage:
 *   insmod apu_probe.ko
 *   dmesg | grep apu_probe
 *   rmmod apu_probe
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/io.h>
#include <linux/delay.h>

/* ============================================================
 * Physical addresses from MT6853/MT6855 device tree and driver
 * ============================================================ */

/* APU RPC (Remote Power Controller) — handles power on/off
 * DT: apu_rpc@190f0000
 * reg[0]: RPC registers */
#define APU_RPC_BASE        0x190F0000
#define APU_RPC_SIZE        0x1000

/* SPM (System Power Manager) — controls APU power domain
 * DT: reg[2] of apu_rpc node
 * Used for buck isolation and cross-wake */
#define SPM_BASE            0x10006000
#define SPM_SIZE            0x1000

/* Top clock config — for APU clock mux
 * DT: reg[1] of apu_rpc node  
 * syscon@10000000 */
#define TOPCK_BASE          0x10000000
#define TOPCK_SIZE          0x1000

/* APU CONN config — APU connection/bus config
 * DT: syscon@19020000 */
#define APU_CONN_BASE       0x19020000
#define APU_CONN_SIZE       0x1000

/* APU VCORE clock — APU vcore config
 * DT: syscon@19029000 */
#define APU_VCORE_BASE      0x19029000
#define APU_VCORE_SIZE      0x1000

/* MDLA core 0 — the actual DLA engine
 * Address from MT6855 MDLA driver (MDLA v2.0) */
#define APU_MDLA0_BASE      0x19034000
#define APU_MDLA0_SIZE      0x1000

/* APU IOMMU
 * DT: iommu@19010000 */
#define APU_IOMMU_BASE      0x19010000
#define APU_IOMMU_SIZE      0x1000

/* ============================================================
 * RPC register offsets (from apu_rpc.h)
 * ============================================================ */
#define RPC_TOP_CON         0x000
#define RPC_TOP_SEL         0x004
#define RPC_SW_FIFO_WE      0x008
#define RPC_INTF_PWR_RDY    0x044
#define RPC_SW_TYPE0        0x200  /* APUTOP */
#define RPC_SW_TYPE6        0x260  /* MDLA0 */

/* SPM register offsets */
#define SPM_OTHER_PWR_STATUS    0x178
#define SPM_BUCK_ISOLATION      0x39C
#define SPM_CROSS_WAKE_M01_REQ  0x670

/* RPC wake IDs */
#define RPC_TOP_WAKE_ID     1
#define RPC_MDLA0_WAKE_ID   6

/* ============================================================
 * MDLA register offsets (from mdla_hw_cmde_v2_0.h)
 * ============================================================ */
#define MDLA_CG_CON         0x000
#define MDLA_SW_RST         0x00C
#define MREG_TOP_G_INTP0    0x0504
#define MREG_TOP_G_CDMA0    0x0510
#define MREG_TOP_ENG0       0x0550
#define CFG_PMCR            0x0E00
#define PMU_CLK_CNT         0x0E04

/* ============================================================
 * Module implementation
 * ============================================================ */

static void __iomem *rpc_base;
static void __iomem *spm_base;
static void __iomem *conn_base;
static void __iomem *vcore_base;
static void __iomem *mdla_base;
static void __iomem *iommu_base;

/* Safe register read — catches bus errors via read value */
static u32 safe_readl(void __iomem *addr, const char *name, u32 offset)
{
    u32 val = readl(addr + offset);
    pr_info("[apu_probe] %s + 0x%03x = 0x%08x\n", name, offset, val);
    return val;
}

/* Probe a register region — try to map and read signature registers */
static void __iomem *probe_region(phys_addr_t phys, size_t size,
                                   const char *name)
{
    void __iomem *base;

    pr_info("[apu_probe] Mapping %s at 0x%09llx (size 0x%lx)\n",
            name, (u64)phys, (unsigned long)size);

    base = ioremap(phys, size);
    if (!base) {
        pr_err("[apu_probe] FAILED to map %s at 0x%09llx\n",
               name, (u64)phys);
        return NULL;
    }

    pr_info("[apu_probe] Mapped %s at VA %px\n", name, base);
    return base;
}

static int __init apu_probe_init(void)
{
    u32 val;
    int alive = 0;

    pr_info("[apu_probe] ========================================\n");
    pr_info("[apu_probe] Moltar APU Hardware Probe v1.0\n");
    pr_info("[apu_probe] Target: MT6855 / Dimensity 930\n");
    pr_info("[apu_probe] ========================================\n");

    /* Step 1: Map SPM registers (always accessible) */
    spm_base = probe_region(SPM_BASE, SPM_SIZE, "SPM");
    if (spm_base) {
        pr_info("[apu_probe] --- SPM Status ---\n");
        safe_readl(spm_base, "SPM", SPM_OTHER_PWR_STATUS);
        safe_readl(spm_base, "SPM", SPM_BUCK_ISOLATION);
        safe_readl(spm_base, "SPM", SPM_CROSS_WAKE_M01_REQ);

        /* Check if APU power domain bit is set in SPM */
        val = readl(spm_base + SPM_OTHER_PWR_STATUS);
        pr_info("[apu_probe] APU power domain (bit 5): %s\n",
                (val & (1 << 5)) ? "ON" : "OFF");
    }

    /* Step 2: Map APU RPC registers */
    rpc_base = probe_region(APU_RPC_BASE, APU_RPC_SIZE, "APU_RPC");
    if (rpc_base) {
        pr_info("[apu_probe] --- RPC Registers ---\n");
        safe_readl(rpc_base, "RPC", RPC_TOP_CON);
        safe_readl(rpc_base, "RPC", RPC_TOP_SEL);
        safe_readl(rpc_base, "RPC", RPC_INTF_PWR_RDY);
        safe_readl(rpc_base, "RPC", RPC_SW_TYPE0);
        safe_readl(rpc_base, "RPC", RPC_SW_TYPE6);

        /* Test if RPC is alive by writing to dummy register bits */
        val = readl(rpc_base + RPC_TOP_SEL);
        writel(val | (0x3a << 26), rpc_base + RPC_TOP_SEL);
        alive = ((readl(rpc_base + RPC_TOP_SEL) >> 26) & 0x3f) == 0x3a;
        /* Clear test bits */
        writel(val, rpc_base + RPC_TOP_SEL);
        pr_info("[apu_probe] RPC alive test: %s\n",
                alive ? "ALIVE" : "NOT RESPONDING");
    }

    /* Step 3: Map APU CONN */
    conn_base = probe_region(APU_CONN_BASE, APU_CONN_SIZE, "APU_CONN");
    if (conn_base) {
        pr_info("[apu_probe] --- APU CONN ---\n");
        safe_readl(conn_base, "CONN", 0x000);  /* CG_CON */
        safe_readl(conn_base, "CONN", 0x004);  /* CG_SET */
        safe_readl(conn_base, "CONN", 0x008);  /* CG_CLR */
    }

    /* Step 4: Map APU VCORE */
    vcore_base = probe_region(APU_VCORE_BASE, APU_VCORE_SIZE, "APU_VCORE");
    if (vcore_base) {
        pr_info("[apu_probe] --- APU VCORE ---\n");
        safe_readl(vcore_base, "VCORE", 0x000);
        safe_readl(vcore_base, "VCORE", 0x004);
    }

    /* Step 5: Map APU IOMMU */
    iommu_base = probe_region(APU_IOMMU_BASE, APU_IOMMU_SIZE, "APU_IOMMU");
    if (iommu_base) {
        pr_info("[apu_probe] --- APU IOMMU ---\n");
        safe_readl(iommu_base, "IOMMU", 0x000);
        safe_readl(iommu_base, "IOMMU", 0x020);  /* Fault addr */
        safe_readl(iommu_base, "IOMMU", 0x024);  /* Fault status */
    }

    /* Step 6: Map MDLA */
    mdla_base = probe_region(APU_MDLA0_BASE, APU_MDLA0_SIZE, "MDLA0");
    if (mdla_base) {
        pr_info("[apu_probe] --- MDLA0 ---\n");
        safe_readl(mdla_base, "MDLA", MDLA_CG_CON);
        safe_readl(mdla_base, "MDLA", MDLA_SW_RST);
        safe_readl(mdla_base, "MDLA", MREG_TOP_G_INTP0);
        safe_readl(mdla_base, "MDLA", MREG_TOP_ENG0);
        safe_readl(mdla_base, "MDLA", CFG_PMCR);
    }

    pr_info("[apu_probe] ========================================\n");
    pr_info("[apu_probe] Probe complete. Check results above.\n");
    pr_info("[apu_probe] ========================================\n");

    return 0;
}

static void __exit apu_probe_exit(void)
{
    if (mdla_base)
        iounmap(mdla_base);
    if (iommu_base)
        iounmap(iommu_base);
    if (vcore_base)
        iounmap(vcore_base);
    if (conn_base)
        iounmap(conn_base);
    if (rpc_base)
        iounmap(rpc_base);
    if (spm_base)
        iounmap(spm_base);

    pr_info("[apu_probe] Module unloaded, all regions unmapped.\n");
}

module_init(apu_probe_init);
module_exit(apu_probe_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Moltar Project");
MODULE_DESCRIPTION("APU Hardware Probe for MT6855/Dimensity 930");
