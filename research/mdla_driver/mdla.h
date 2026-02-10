/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Moltar MDLA 2.x Driver - Minimal userspace interface
 *
 * Direct hardware access to MediaTek Deep Learning Accelerator
 * on MT6855/Dimensity 930 (MDLA 2.0/2.1)
 *
 * Author: Moltar Project
 */

#ifndef _MOLTAR_MDLA_H_
#define _MOLTAR_MDLA_H_

#include <stdint.h>
#include <stdbool.h>

/* Physical addresses from MT6855 */
#define MDLA_BASE          0x19034000
#define MDLA_SIZE          0x1000

#define SPM_BASE           0x10006000
#define SPM_SIZE           0x1000

#define APU_RPC_BASE      0x190F0000
#define APU_RPC_SIZE      0x1000

#define APU_IOMMU_BASE    0x19010000
#define APU_IOMMU_SIZE    0x1000

/* MDLA registers (relative to MDLA_BASE) */
#define REG_MDLA_CG_CON       0x000
#define REG_MDLA_CG_SET       0x004
#define REG_MDLA_CG_CLR       0x008
#define REG_MDLA_SW_RST       0x00C
#define REG_MDLA_CTRL         0x110
#define REG_MDLA_CSYSREQ      0x114
#define REG_MDLA_CSYSACK      0x118

/* Command interface registers */
#define REG_MREG_TOP_G_INTP0  0x0504
#define REG_MREG_TOP_G_CDMA0  0x0510
#define REG_MREG_TOP_G_CDMA1  0x0514
#define REG_MREG_TOP_G_CDMA2  0x0518
#define REG_MREG_TOP_G_CDMA3  0x051C
#define REG_MREG_TOP_G_CDMA4  0x0520
#define REG_MREG_TOP_G_CDMA5  0x0524
#define REG_MREG_TOP_G_CDMA6  0x0528
#define REG_MREG_TOP_G_FIN0   0x0534
#define REG_MREG_TOP_G_FIN1   0x0538
#define REG_MREG_TOP_G_FIN3   0x0584
#define REG_MREG_TOP_G_FIN4   0x058C
#define REG_MREG_TOP_G_IDLE   0x0544
#define REG_MREG_TOP_G_ENG0   0x0550

/* SPM registers */
#define SPM_OTHER_PWR_STATUS 0x178

/* Bit definitions */
#define MDLA_SW_MDLA_RST_MASK  BIT(0)
#define MDLA_SW_APB_RST_MASK   BIT(6)
#define MDLA_SW_RST_MASK      (MDLA_SW_MDLA_RST_MASK | MDLA_SW_APB_RST_MASK)

#define INTR_SWCMD_DONE        BIT(2)
#define INTR_SWCMD_TILECNT_INT BIT(1)
#define INTR_SUPPORT_MASK      (INTR_SWCMD_DONE | BIT(21) /* FIN4_CMD_STOP */)

#define APU_PWR_BIT           BIT(5)  /* Bit 5 in SPM_OTHER_PWR_STATUS */

typedef struct {
    int fd;
    void *mdla_base;
    void *spm_base;
    void *rpc_base;
    void *iommu_base;
} mdla_t;

/* Initialization */
int mdla_open(mdla_t *dev);
void mdla_close(mdla_t *dev);

/* Power management */
int mdla_power_on(mdla_t *dev);
int mdla_power_off(mdla_t *dev);
bool mdla_is_powered(mdla_t *dev);

/* Reset */
int mdla_reset(mdla_t *dev);

/* Status */
uint32_t mdla_read_status(mdla_t *dev);
bool mdla_is_idle(mdla_t *dev);
uint32_t mdla_read_finish(mdla_t *dev);

/* Command submission */
int mdla_write_cmd(mdla_t *dev, uint32_t offset, uint32_t value);
uint32_t mdla_read_cmd(mdla_t *dev, uint32_t offset);

/* Interrupt handling */
uint32_t mdla_read_intr(mdla_t *dev);
void mdla_clear_intr(mdla_t *dev, uint32_t mask);

/* Utility */
void mdla_dump_regs(mdla_t *dev);

#endif /* _MOLTAR_MDLA_H_ */
