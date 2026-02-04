# Root Moto G Power 5G 2023 - Quick Guide

## The Secret: Use FastbootD, Not Regular Fastboot

Motorola blocks modified boot images in regular fastboot with "Preflash validation failed".

**FastbootD (userspace fastboot) has NO validation.**

## Commands

```bash
# 1. Patch boot.img with Magisk app on phone first, then:

# 2. Reboot to FastbootD (NOT regular fastboot!)
adb reboot fastboot

# 3. Verify you're in userspace mode
fastboot getvar is-userspace
# Must show: is-userspace: yes

# 4. Flash - no validation error!
fastboot flash boot magisk_patched-XXXXX.img

# 5. Reboot and enjoy root
fastboot reboot
```

## Verify Root
```bash
adb shell "su -c id"
# uid=0(root) gid=0(root) context=u:r:magisk:s0
```

## Key Difference

| Mode | Command to Enter | Validation |
|------|------------------|------------|
| Regular Fastboot | `adb reboot bootloader` | ❌ BLOCKED |
| FastbootD | `adb reboot fastboot` | ✅ WORKS |

That's it. The billion-dollar security bypass is one word: `fastboot` instead of `bootloader`.

---

Tested: Moto G Power 5G 2023 (XT2311-4), Android 14, MT6855
Works on: Likely all 2023+ Motorola with FastbootD support

**It's your phone. Root it.**
