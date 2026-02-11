/*
 * mdla_read_main.c - Unmask APU resources + full cross-wake retry
 *
 * Strategy:
 *   1. Unmask all APU resource requests in SPM_SRC_MASK_0 (bits 5-9)
 *   2. Release buck isolation
 *   3. Configure SPM2APU_CON
 *   4. Set WAKEUP_APU
 *   5. Monitor MD32 PC, all status registers, and APU address space
 *   6. Try alternate approaches if cross-wake still fails
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Unmask APU resources + cross-wake retry for MT6855");

/* Base addresses */
#define SPM_BASE          0x1C001000UL
#define APU_RPC_BASE      0x190F0000UL
#define APU_CONN_BASE     0x19020000UL
#define MDLA0_BASE        0x19034000UL
#define REG_SIZE          0x1000

/* SPM registers */
#define MD32PCM_PC              0x194
#define SPM_OTHER_PWR_STATUS    0x178
#define SPM_SRC_MASK_0          0x81C
#define SPM_SRC_MASK_1          0x820
#define SPM_SRC_MASK_2          0x824
#define SPM_REQ_STA_0           0x838
#define SPM2APU_CON             0x414
#define SOC_BUCK_ISO_CON        0xF30
#define SPM_CROSS_WAKE_M01_REQ  0x670
#define PWR_STATUS              0xF34
#define XPU_PWR_STATUS          0xF3C

/* RPC registers */
#define RPC_TOP_CON         0x000
#define RPC_TOP_SEL         0x004
#define RPC_SW_FIFO_WE      0x008
#define RPC_STATUS          0x014
#define RPC_INTF_PWR_RDY    0x044
#define RPC_SW_TYPE0        0x200
#define RPC_SW_TYPE6        0x260

/* APU resource mask bits in SPM_SRC_MASK_0 */
#define APU_APSRC_MASK_B    BIT(5)
#define APU_DDREN_MASK_B    BIT(6)
#define APU_INFRA_MASK_B    BIT(7)
#define APU_SRCCLKENA_MASK_B BIT(8)
#define APU_VRF18_MASK_B    BIT(9)
#define APU_ALL_MASKS       (APU_APSRC_MASK_B | APU_DDREN_MASK_B | \
			     APU_INFRA_MASK_B | APU_SRCCLKENA_MASK_B | \
			     APU_VRF18_MASK_B)

/* SPM2APU_CON bits */
#define SPM2APU_RPC_SRAM_MUX    BIT(0)
#define SPM2APU_VCORE_OFF_ISO   BIT(1)
#define SPM2APU_ARE_REQ         BIT(4)
#define SPM2APU_ARE_ACK         BIT(8)
#define SPM2APU_ACTIVE          BIT(9)

/* SOC_BUCK_ISO_CON bits */
#define VAPU_EXT_BUCK_ISO       BIT(4)
#define AOC_VAPU_SRAM_ISO       BIT(5)
#define AOC_VAPU_SRAM_LATCH     BIT(6)
#define AOC_VAPU_ANA_ISO        BIT(7)
#define VAPU_ALL_ISO            (VAPU_EXT_BUCK_ISO | AOC_VAPU_SRAM_ISO | \
				 AOC_VAPU_SRAM_LATCH | AOC_VAPU_ANA_ISO)

#define WAKEUP_APU  BIT(0)

static void __iomem *spm_base;
static void __iomem *rpc_base;
static u32 orig_src_mask_0;

static inline void apu_setl(u32 bits, void __iomem *addr)
{
	writel(readl(addr) | bits, addr);
}

static inline void apu_clearl(u32 bits, void __iomem *addr)
{
	writel(readl(addr) & ~bits, addr);
}

static noinline void scan_apu(const char *label)
{
	void __iomem *v;
	u32 val;
	int i, count;

	/* Quick check of RPC */
	pr_err("MDLA_PROBE: [%s] RPC_CON=0x%08x SEL=0x%08x STA=0x%08x RDY=0x%08x\n",
	       label,
	       readl(rpc_base + RPC_TOP_CON),
	       readl(rpc_base + RPC_TOP_SEL),
	       readl(rpc_base + RPC_STATUS),
	       readl(rpc_base + RPC_INTF_PWR_RDY));

	/* Quick check of APU_CONN first 16 regs */
	v = ioremap(APU_CONN_BASE, 0x40);
	if (v) {
		count = 0;
		for (i = 0; i < 0x40; i += 4) {
			val = readl(v + i);
			if (val != 0) count++;
		}
		pr_err("MDLA_PROBE: [%s] APU_CONN first 64B: %d non-zero\n",
		       label, count);
		iounmap(v);
	}
}

static int __init mdla_read_init(void)
{
	u32 val;
	int i;

	pr_err("MDLA_PROBE: ========== START (unmask APU + cross-wake) ==========\n");

	spm_base = ioremap(SPM_BASE, 0x2000);
	if (!spm_base) {
		pr_err("MDLA_PROBE: FAIL ioremap SPM\n");
		return 0;
	}
	rpc_base = ioremap(APU_RPC_BASE, REG_SIZE);
	if (!rpc_base) {
		pr_err("MDLA_PROBE: FAIL ioremap RPC\n");
		iounmap(spm_base);
		return 0;
	}

	/* ===== Baseline ===== */
	pr_err("MDLA_PROBE: --- Baseline ---\n");
	pr_err("MDLA_PROBE: SRC_MASK_0=0x%08x SPM2APU=0x%08x BUCK_ISO=0x%08x\n",
	       readl(spm_base + SPM_SRC_MASK_0),
	       readl(spm_base + SPM2APU_CON),
	       readl(spm_base + SOC_BUCK_ISO_CON));
	pr_err("MDLA_PROBE: REQ_STA_0=0x%08x OTHER_PWR=0x%08x XPU_PWR=0x%08x\n",
	       readl(spm_base + SPM_REQ_STA_0),
	       readl(spm_base + SPM_OTHER_PWR_STATUS),
	       readl(spm_base + XPU_PWR_STATUS));
	scan_apu("BASELINE");

	/* ===== Step 1: Unmask ALL APU resource requests ===== */
	pr_err("MDLA_PROBE: Step 1: Unmask APU resource requests\n");
	orig_src_mask_0 = readl(spm_base + SPM_SRC_MASK_0);
	apu_setl(APU_ALL_MASKS, spm_base + SPM_SRC_MASK_0);
	pr_err("MDLA_PROBE: SRC_MASK_0: 0x%08x -> 0x%08x\n",
	       orig_src_mask_0, readl(spm_base + SPM_SRC_MASK_0));
	udelay(100);

	/* ===== Step 2: Release buck isolation ===== */
	pr_err("MDLA_PROBE: Step 2: Release VAPU buck isolation\n");
	apu_clearl(VAPU_ALL_ISO, spm_base + SOC_BUCK_ISO_CON);
	udelay(100);
	pr_err("MDLA_PROBE: BUCK_ISO after = 0x%08x\n",
	       readl(spm_base + SOC_BUCK_ISO_CON));

	/* ===== Step 3: Configure SPM2APU_CON ===== */
	pr_err("MDLA_PROBE: Step 3: Configure SPM2APU_CON\n");
	apu_setl(SPM2APU_RPC_SRAM_MUX, spm_base + SPM2APU_CON);
	apu_clearl(SPM2APU_VCORE_OFF_ISO, spm_base + SPM2APU_CON);
	apu_setl(SPM2APU_ARE_REQ, spm_base + SPM2APU_CON);
	pr_err("MDLA_PROBE: SPM2APU = 0x%08x\n",
	       readl(spm_base + SPM2APU_CON));

	/* ===== Step 4: Set WAKEUP_APU ===== */
	pr_err("MDLA_PROBE: Step 4: Set WAKEUP_APU + monitor\n");
	apu_setl(WAKEUP_APU, spm_base + SPM_CROSS_WAKE_M01_REQ);

	/* Sample rapidly for 5ms */
	for (i = 0; i < 50; i++) {
		val = readl(spm_base + SPM_OTHER_PWR_STATUS);
		if (val & BIT(5)) {
			pr_err("MDLA_PROBE: *** OTHER_PWR bit5 SET at T+%d! val=0x%08x ***\n",
			       i, val);
			break;
		}
		if (readl(spm_base + SPM2APU_CON) & SPM2APU_ACTIVE) {
			pr_err("MDLA_PROBE: *** SPM2APU ACTIVE set at T+%d! ***\n", i);
			break;
		}
		udelay(100);
	}

	/* Check state after initial wait */
	pr_err("MDLA_PROBE: After 5ms: OTHER_PWR=0x%08x XPU_PWR=0x%08x SPM2APU=0x%08x\n",
	       readl(spm_base + SPM_OTHER_PWR_STATUS),
	       readl(spm_base + XPU_PWR_STATUS),
	       readl(spm_base + SPM2APU_CON));
	pr_err("MDLA_PROBE: MD32PC=0x%08x CROSS_WAKE=0x%08x REQ_STA_0=0x%08x\n",
	       readl(spm_base + MD32PCM_PC),
	       readl(spm_base + SPM_CROSS_WAKE_M01_REQ),
	       readl(spm_base + SPM_REQ_STA_0));

	/* Wait another 50ms */
	for (i = 0; i < 5; i++)
		udelay(10000);

	pr_err("MDLA_PROBE: After 55ms: OTHER_PWR=0x%08x SPM2APU=0x%08x\n",
	       readl(spm_base + SPM_OTHER_PWR_STATUS),
	       readl(spm_base + SPM2APU_CON));
	scan_apu("POST_WAKE");

	/* ===== Step 5: Try RPC init regardless ===== */
	pr_err("MDLA_PROBE: Step 5: Try RPC init anyway\n");
	writel(0xFF, rpc_base + RPC_SW_TYPE0);
	writel(0x03, rpc_base + RPC_SW_TYPE6);
	val = 0x1800501E;
	writel(val, rpc_base + RPC_TOP_SEL);
	pr_err("MDLA_PROBE: RPC_TOP_SEL readback = 0x%08x (wrote 0x%08x)\n",
	       readl(rpc_base + RPC_TOP_SEL), val);

	/* Try wake via FIFO */
	writel(0x11, rpc_base + RPC_SW_FIFO_WE);  /* APUTOP | BIT(4) */
	udelay(1000);
	pr_err("MDLA_PROBE: After FIFO write: RDY=0x%08x STA=0x%08x\n",
	       readl(rpc_base + RPC_INTF_PWR_RDY),
	       readl(rpc_base + RPC_STATUS));

	/* ===== Step 6: Try directly writing PWR_ON bits to each known-unused register ===== */
	/* There's no APU_PWR_CON, but let's try writing PWR_ON + PWR_ON_2ND
	 * to the ADSP_TOP_PWR_CON (0xE18) which is currently OFF and adjacent
	 * to the audio domain. ADSP shares infrastructure with APU on some SoCs.
	 * Reading first to confirm it's powered off, then we'll try it.
	 * NOTE: This is speculative - ADSP is NOT APU, but some MT6855 variants
	 * may route APU through the ADSP power domain path. */
	pr_err("MDLA_PROBE: Step 6: Check ADSP domains\n");
	pr_err("MDLA_PROBE: ADSP_TOP_PWR (0xE18) = 0x%08x\n",
	       readl(spm_base + 0xE18));
	pr_err("MDLA_PROBE: ADSP_INFRA   (0xE1C) = 0x%08x\n",
	       readl(spm_base + 0xE1C));
	pr_err("MDLA_PROBE: ADSP_AO      (0xE20) = 0x%08x\n",
	       readl(spm_base + 0xE20));

	/* ===== Step 7: Scan wider SPM range for any change ===== */
	pr_err("MDLA_PROBE: Step 7: Check for any SPM status changes\n");
	pr_err("MDLA_PROBE: PWR_STA=0x%08x PWR_STA2=0x%08x\n",
	       readl(spm_base + PWR_STATUS),
	       readl(spm_base + 0xF38));
	pr_err("MDLA_PROBE: XPU_STA=0x%08x XPU_STA2=0x%08x\n",
	       readl(spm_base + XPU_PWR_STATUS),
	       readl(spm_base + 0xF40));

	/* Check all REQ_STA for any APU activity */
	for (i = 0; i < 5; i++) {
		val = readl(spm_base + SPM_REQ_STA_0 + i * 4);
		pr_err("MDLA_PROBE: REQ_STA_%d = 0x%08x\n", i, val);
	}

	/* ===== Cleanup ===== */
	pr_err("MDLA_PROBE: Cleanup: restore masks and isolation\n");

	/* Clear wakeup */
	apu_clearl(WAKEUP_APU, spm_base + SPM_CROSS_WAKE_M01_REQ);

	/* Restore original mask */
	writel(orig_src_mask_0, spm_base + SPM_SRC_MASK_0);
	pr_err("MDLA_PROBE: SRC_MASK_0 restored to 0x%08x\n",
	       readl(spm_base + SPM_SRC_MASK_0));

	/* Restore buck isolation */
	apu_setl(VAPU_ALL_ISO, spm_base + SOC_BUCK_ISO_CON);

	iounmap(rpc_base);
	rpc_base = NULL;
	iounmap(spm_base);
	spm_base = NULL;

	pr_err("MDLA_PROBE: ========== DONE ==========\n");
	return 0;
}

static void __exit mdla_read_exit(void)
{
	if (rpc_base) { iounmap(rpc_base); rpc_base = NULL; }
	if (spm_base) { iounmap(spm_base); spm_base = NULL; }
	pr_err("MDLA_PROBE: UNLOADED\n");
}

module_init(mdla_read_init);
module_exit(mdla_read_exit);
