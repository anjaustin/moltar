# Minimal Boot Options for Moto G Power 5G (2023)

## Goal

Transform the phone from a full Android device into a minimal Linux-like system that boots directly to a terminal/TriX interface, maximizing RAM for AI workloads.

---

## Option 1: Headless Mode (Disable SystemUI)

**Difficulty:** Easy
**RAM Savings:** ~250MB
**Reversibility:** Easy (via ADB)

This disables the entire Android UI, making the phone ADB/SSH-only.

### Enable Headless Mode
```bash
# Disable SystemUI (no more screen UI)
adb shell "su -c 'pm disable-user --user 0 com.android.systemui'"

# Disable launcher
adb shell "su -c 'pm disable-user --user 0 com.motorola.launcher3'"

# The screen will go black, but the phone is still running
# Access via ADB or SSH (if configured)
```

### Important: SystemUI Respawns!

Simply disabling SystemUI isn't enough - `system_server` will keep respawning it.
To truly disable it, you need one of these approaches:

**Option A: Magisk Module**
Create a module that moves SystemUI APK out of the way:
```bash
# In Magisk module's post-fs-data.sh
mv /system/priv-app/SystemUIGoogle /system/priv-app/SystemUIGoogle.bak
```

**Option B: Rename the APK (requires remounting /system)**
```bash
adb shell "su -c 'mount -o remount,rw /system'"
adb shell "su -c 'mv /system/priv-app/SystemUIGoogle /system/priv-app/SystemUIGoogle.bak'"
adb reboot
```

**Option C: Use a "debloater" Magisk module**
- Install "Debloater" module from Magisk repo
- Add SystemUI to the disable list

### Access the Headless Device
```bash
# Via ADB
adb shell

# Via SSH (if Termux SSHD is running)
ssh user@<phone-ip> -p 8022
```

### Re-enable UI
```bash
adb shell "su -c 'pm enable com.android.systemui'"
adb shell "su -c 'pm enable com.motorola.launcher3'"
adb reboot
```

### Auto-start Services in Headless Mode

Edit Termux boot script (`~/.termux/boot/start_trix.sh`):
```bash
#!/data/data/com.termux/files/usr/bin/bash

# Start SSHD for remote access
sshd

# Start llama-server
export LD_LIBRARY_PATH=$HOME/noprofile
taskset 0x80 $HOME/noprofile/llama-server \
    -m $HOME/LFM2-1.2B-Q4_0.gguf \
    -t 1 --host 0.0.0.0 --port 8080 \
    -c 3840 -np 1 --no-mmap \
    > $HOME/server.log 2>&1 &

# Optional: Start Qdrant
# $HOME/qdrant &
```

---

## Option 2: Kiosk Mode (Single App)

**Difficulty:** Medium
**RAM Savings:** ~150MB (compared to full Android)
**Reversibility:** Easy

Lock the device to a single app (like Termux or a custom TriX app).

### Using ADB
```bash
# Set Termux as device owner (requires factory reset first on some devices)
adb shell dpm set-device-owner com.termux/.app.TermuxDeviceAdminReceiver

# Or use lock task mode
adb shell "su -c 'am start-activity --activity-single-top -n com.termux/.app.TermuxActivity'"
adb shell "su -c 'settings put global device_provisioned 0'"
```

### Using Magisk Module
Create a module that:
1. Disables setup wizard
2. Sets default launcher to a minimal app
3. Hides navigation bar

---

## Option 3: Replace SystemUI with Custom App

**Difficulty:** Hard
**RAM Savings:** ~200MB
**Reversibility:** Medium (need to flash stock)

Replace the entire SystemUI with a minimal custom implementation.

### Steps
1. Build a minimal SystemUI replacement APK
2. Sign with platform key (or use Magisk to bypass)
3. Replace `/system/priv-app/SystemUIGoogle/` or similar

### Minimal SystemUI Features
- Black screen with status bar only
- Termux widget for quick access
- No recent apps, no notification shade
- Hardware buttons still work

---

## Option 4: Init.d Script (Boot Directly to Shell)

**Difficulty:** Hard
**RAM Savings:** Maximum
**Reversibility:** Hard

Modify the init process to skip Android entirely.

### Concept
Edit `/system/etc/init/` scripts to:
1. Skip zygote (Android runtime)
2. Start only essential services
3. Launch a shell or custom daemon

### Warning
This essentially breaks Android and turns the device into a pure Linux box.
You lose: touchscreen (needs Android input system), display (needs SurfaceFlinger), etc.

### What You Keep
- Kernel + all drivers
- USB/ADB access
- Network (with manual configuration)
- Storage access

---

## Option 5: chroot to Full Linux

**Difficulty:** Medium
**RAM Savings:** Depends on what you run
**Reversibility:** Easy

Run a full Linux distro alongside (or instead of) Android userspace.

### Using Linux Deploy
1. Install Linux Deploy from F-Droid
2. Create Debian/Ubuntu/Arch container
3. Configure to start on boot
4. SSH into Linux environment

### Using proot-distro (Termux)
```bash
# Install a distro
pkg install proot-distro
proot-distro install debian

# Login
proot-distro login debian

# Now you're in Debian
apt update && apt install -y build-essential
```

### Limitation
Still runs on top of Android kernel, but gives you full apt/systemd environment.

---

## Option 6: Droidian / Ubuntu Touch Port

**Difficulty:** Very Hard (weeks/months of work)
**RAM Savings:** Potentially significant
**Reversibility:** Can dual-boot

Port a full Linux mobile OS to the device.

### What's Needed
1. Kernel source (MediaTek rarely releases)
2. Device tree configuration
3. Halium compatibility layer
4. libhybris for Android driver access

### Current Status for MT6855
- No existing port
- MediaTek kernel sources are often incomplete
- Would be a significant community project

---

## Recommended Approach for TriX

### Phase 1: Headless + Termux (NOW)
```bash
# 1. Debloat aggressively (done)
bash debloat_moto.sh

# 2. Set up Termux boot script (done)
# ~/.termux/boot/start_trix.sh

# 3. Optional: Go fully headless
adb shell "su -c 'pm disable-user --user 0 com.android.systemui'"
```

### Phase 2: Custom Minimal Launcher
Create a simple Android app that:
- Shows only a terminal (WebView to llama-server UI)
- Has a floating widget for status
- Disables itself when not needed

### Phase 3: Full Linux (Future)
If community interest grows:
- Work on Droidian/Halium port
- Contribute to postmarketOS MT6855 support
- Create minimal Android-free boot option

---

## Memory Comparison

| Configuration | Free RAM | Usable for AI |
|---------------|----------|---------------|
| Stock Android | ~180MB | Barely |
| Debloated | ~300MB | Yes |
| Headless (no UI) | ~550MB | Comfortable |
| chroot Linux | ~400MB | Yes |
| Pure Linux (theoretical) | ~800MB+ | Excellent |

---

## Quick Commands Reference

### Go Headless
```bash
adb shell "su -c 'pm disable-user --user 0 com.android.systemui'"
```

### Restore UI
```bash
adb shell "su -c 'pm enable com.android.systemui'" && adb reboot
```

### Check Free RAM
```bash
adb shell "su -c 'free -m'"
```

### Start Termux via ADB
```bash
adb shell "am start -n com.termux/.app.TermuxActivity"
```

### Run Command in Termux via ADB
```bash
adb shell "su -c 'su -c \"cd /data/data/com.termux/files/home && ./start_server.sh\" - u0_a123'"
```

---

## Files Created

- `debloat_moto.sh` - Automated debloat script
- `setup_termux.sh` - Termux boot configuration
- `~/.termux/boot/start_trix.sh` - Auto-start script (on device)

---

*The goal: A $100 phone that boots in 30 seconds to a ready-to-use AI assistant, using minimal resources.*
