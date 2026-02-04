# How to Root Moto G Power 5G (2023) - The FastbootD Bypass

## TL;DR

**Motorola's preflash validation blocks Magisk in regular fastboot mode. But FastbootD (userspace fastboot) has NO validation. Flash through FastbootD and you win.**

```bash
adb reboot fastboot                      # FastbootD, NOT regular fastboot
fastboot flash boot magisk_patched.img   # No preflash validation!
fastboot reboot                          # Welcome to root
```

---

## The Problem

Motorola devices from 2023+ (especially MediaTek) have multiple layers of protection:

1. **DAA (Download Agent Authentication)** - Blocks mtkclient/BROM exploits
2. **Preflash Validation** - Bootloader checks boot.img hash before flashing
3. **AVB (Android Verified Boot)** - Verifies boot chain integrity

Even with an **unlocked bootloader**, you get this when trying to flash a patched boot.img:

```
FAILED (remote: 'Preflash validation failed')
```

Everyone said it was impossible. They were wrong.

---

## The Discovery

**FastbootD (userspace fastboot) does NOT perform preflash validation.**

- Regular Fastboot = Runs in bootloader = Preflash validation = BLOCKED
- FastbootD = Runs in recovery/userspace = No validation = SUCCESS

This works because FastbootD is a different code path that Motorola didn't lock down.

---

## Device Info (Tested)

| Property | Value |
|----------|-------|
| Device | Moto G Power 5G (2023) |
| Model | XT2311-4 |
| Codename | devonn |
| SoC | MediaTek Dimensity 7020 (MT6855) |
| Android | 14 |
| Build | U1TO34.1-157-5 |
| Bootloader | Unlocked |

---

## Requirements

- Moto G Power 5G (2023) with **unlocked bootloader**
- USB cable
- Computer with ADB/Fastboot installed
- Magisk APK (latest from https://github.com/topjohnwu/Magisk/releases)
- ~30 minutes

---

## Step-by-Step Guide

### Step 1: Unlock Bootloader (if not already done)

1. Enable Developer Options (tap Build Number 7 times)
2. Enable OEM Unlocking in Developer Options
3. `adb reboot bootloader`
4. `fastboot oem unlock` (or use Motorola's unlock site)
5. Confirm on device (THIS WIPES YOUR DATA)

### Step 2: Get Your Stock Boot Image

Option A: Extract from firmware
- Download firmware from https://mirrors.lolinet.com/firmware/lenomola/2023/devonn/official/
- Extract boot.img from the zip

Option B: Dump from device (if already rooted or have TWRP)
```bash
adb shell "su -c 'dd if=/dev/block/by-name/boot_a of=/sdcard/boot.img'"
adb pull /sdcard/boot.img
```

### Step 3: Patch with Magisk

1. Copy boot.img to phone: `adb push boot.img /sdcard/Download/`
2. Install Magisk APK on phone
3. Open Magisk → Install → Select and Patch a File
4. Select the boot.img from Download folder
5. Magisk creates `magisk_patched-XXXXX.img` in Download
6. Pull it back: `adb pull /sdcard/Download/magisk_patched-XXXXX.img`

### Step 4: Flash via FastbootD (THE KEY STEP)

**DO NOT use regular fastboot mode. Use FastbootD.**

```bash
# Reboot to FastbootD (userspace fastboot)
adb reboot fastboot

# Wait for device - screen will show "FastbootD" or Android recovery UI
fastboot devices

# Verify you're in userspace mode
fastboot getvar is-userspace
# Should return: is-userspace: yes

# Flash the patched boot image - NO PREFLASH VALIDATION!
fastboot flash boot magisk_patched-XXXXX.img

# Reboot
fastboot reboot
```

### Step 5: Verify Root

```bash
adb shell "su -c id"
```

Expected output:
```
uid=0(root) gid=0(root) groups=0(root) context=u:r:magisk:s0
```

**You now have root.**

---

## Why This Works

### Regular Fastboot (Bootloader Mode)
```
┌─────────────────────────────────────────┐
│           BOOTLOADER                    │
│  ┌─────────────────────────────────┐    │
│  │   Preflash Validation Check     │    │
│  │   - Verify boot.img signature   │    │
│  │   - Check hash against allowed  │    │
│  │   - REJECT if modified          │ ←── BLOCKED
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### FastbootD (Userspace Mode)
```
┌─────────────────────────────────────────┐
│           RECOVERY/USERSPACE            │
│  ┌─────────────────────────────────┐    │
│  │   FastbootD Service             │    │
│  │   - Direct partition write      │    │
│  │   - No preflash validation      │    │
│  │   - Just write the bytes        │ ←── SUCCESS!
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## Troubleshooting

### "fastboot devices" shows nothing in FastbootD
- Try a different USB cable/port
- Make sure you're in FastbootD (screen shows it), not regular fastboot
- On Linux, you may need udev rules

### Phone bootloops after flashing
1. Boot to FastbootD: hold Vol Down during boot, select "FastbootD" or use `adb reboot fastboot`
2. Flash stock boot.img: `fastboot flash boot boot.img`
3. Reboot and try again with a fresh Magisk patch

### Magisk app shows "Requires Additional Setup"
- Open Magisk app
- Follow prompts to complete installation
- May require one more reboot

### dm-verity error on boot
- Boot anyway (press power button)
- Device should boot with orange "unlocked" warning
- If persistent bootloop, flash stock boot.img and try disabling verity:
  ```bash
  fastboot flash vbmeta --disable-verity --disable-verification vbmeta.img
  ```

---

## What DOESN'T Work

We tried everything before discovering FastbootD:

| Method | Result |
|--------|--------|
| Regular fastboot flash | ❌ Preflash validation failed |
| mtkclient BROM mode | ❌ DAA_SIG_VERIFY_FAILED |
| mtkclient with V6 loader | ❌ DAA blocks all loaders |
| fastboot oem fb_mode_set | ❌ Still validates |
| Test point / hardware | ❌ DAA enabled in eFuses |

**FastbootD is the only software-only method that works.**

---

## Applies To (Likely)

This method should work on any Motorola device that:
- Has an unlockable bootloader
- Has FastbootD support (most Android 10+ devices)
- Uses preflash validation (2023+ Motorola devices)

Potentially affected devices:
- Moto G Power 5G (2023) ✅ CONFIRMED
- Moto G Stylus 5G (2023) - Likely
- Moto G 5G (2023) - Likely
- Moto Edge (2023) - Likely
- Other 2023+ Motorola MediaTek devices - Likely

**If you test this on another device, please report back!**

---

## Credits

- Discovered while trying to run on-device AI (TriX/Moltar project)
- FastbootD bypass found after exhausting all other options
- Fuck artificial restrictions on hardware you own

---

## Legal

This guide is for educational purposes. You own your device. Root it.

Unlocking bootloader and rooting may void warranty. 
Flashing incorrect images can brick your device.
You are responsible for your own actions.

---

## Share This

If this helped you, share it everywhere:
- XDA Forums
- Reddit r/MotoG, r/Android, r/androidroot
- Motorola forums
- Anywhere people are told "it's impossible"

**It's YOUR phone. Not Motorola's.**

---

*Last updated: February 4, 2026*
*Tested on: Moto G Power 5G 2023, Android 14, Build U1TO34.1-157-5*
