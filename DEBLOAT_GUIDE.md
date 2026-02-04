# Debloating Moto G Power 5G (2023) for Maximum Performance

## Overview

Stock Android on the Moto G Power 5G 2023 uses **~3.2GB of 3.6GB RAM** just sitting idle. After aggressive debloating, we reduced this significantly, freeing up memory for on-device AI workloads.

**Before debloat:** ~180MB free RAM
**After debloat:** ~270MB+ free RAM

This guide disables 147 packages including Google Play Services, carrier bloat, and Motorola apps.

---

## Prerequisites

- **Rooted device** (see ROOT_GUIDE_MOTO_G_POWER_5G_2023.md)
- ADB installed on computer
- USB debugging enabled

---

## Quick Start

```bash
# Run the debloat script
bash debloat_moto.sh

# Reboot to apply
adb reboot

# Verify memory improvement
adb shell "su -c 'free -m'"
```

---

## What Gets Disabled

### Google Services (~400MB RAM savings)
These are the biggest memory hogs:

| Package | RAM Usage | Description |
|---------|-----------|-------------|
| com.google.android.gms | ~200MB | Google Play Services |
| com.google.android.gms.persistent | ~200MB | GMS background process |
| com.google.android.gsf | ~50MB | Google Services Framework |

**Warning:** Disabling GMS breaks:
- Play Store
- Google app logins (use app-specific passwords or alternatives)
- Push notifications for some apps
- Google Pay, Maps, etc.

### Carrier Bloat
```
com.xfinitymobile.cometcarrierservice
com.pocketgeek.xfinity.android
com.verizon.loginengine.unbranded
com.tmobile.* (multiple)
com.tracfone.* (multiple)
com.metro* (multiple)
```

### Motorola Bloat
```
com.motorola.launcher3          # Home screen (install a lighter one)
com.motorola.timeweatherwidget  # Weather widget
com.motorola.camera3            # Camera (use Open Camera)
com.motorola.personalize        
com.motorola.moto
com.motorola.gamemode
com.motorola.help
com.motorola.demo
```

### Pre-installed Apps
```
com.google.android.youtube
com.google.android.apps.photos
com.google.android.apps.maps
com.google.android.apps.docs
com.facebook.* (all)
com.inmobi.weather             # This weather app uses 150MB!
```

---

## Manual Debloat Commands

### Disable a single package
```bash
adb shell "su -c 'pm disable-user --user 0 <package.name>'"
```

### Re-enable a package
```bash
adb shell "su -c 'pm enable <package.name>'"
```

### List all disabled packages
```bash
adb shell "pm list packages -d"
```

### Check memory usage
```bash
adb shell "su -c 'free -m'"
```

### Find top memory consumers
```bash
adb shell "su -c 'ps -A -o pid,rss,name --sort=-rss | head -20'"
```

---

## Full Disabled Package List (147 packages)

```
com.amazon.appmanager
com.android.chrome
com.android.contacts
com.android.dialer
com.android.vending
com.aura.oobe.motorola
com.aura.services.tmobile
com.facebook.appmanager
com.facebook.services
com.facebook.system
com.glance.lockscreenM
com.google.android.accessibility.switchaccess
com.google.android.adservices.api
com.google.android.apps.carrier.carrierwifi
com.google.android.apps.chromecast.app
com.google.android.apps.docs
com.google.android.apps.docs.editors.docs
com.google.android.apps.docs.editors.slides
com.google.android.apps.fitness
com.google.android.apps.magazines
com.google.android.apps.maps
com.google.android.apps.messaging
com.google.android.apps.nbu.files
com.google.android.apps.photos
com.google.android.apps.restore
com.google.android.apps.tachyon
com.google.android.apps.turbo
com.google.android.apps.walletnfcrel
com.google.android.apps.wellbeing
com.google.android.apps.youtube.music
com.google.android.as
com.google.android.calculator
com.google.android.calendar
com.google.android.contacts
com.google.android.deskclock
com.google.android.dialer
com.google.android.feedback
com.google.android.gm
com.google.android.gms
com.google.android.gms.supervision
com.google.android.googlequicksearchbox
com.google.android.gsf
com.google.android.ims
com.google.android.marvin.talkback
com.google.android.ondevicepersonalization.services
com.google.android.printservice.recommendation
com.google.android.projection.gearhead
com.google.android.videos
com.google.android.youtube
com.google.ar.core
com.handmark.expressweather
com.inmobi.weather
com.ironsrc.aura.appmanager.tmo
com.ironsrc.aura.tmo
com.metro.minus1
com.metropcs.metrozone
com.motorola.actions
com.motorola.aiservices
com.motorola.android.fota
com.motorola.android.launcher.overlay.*
com.motorola.android.provisioning
com.motorola.att.phone.extensions
com.motorola.audiorecorder
com.motorola.brapps
com.motorola.camera3
com.motorola.carrierconfig
com.motorola.carriersettingsext
com.motorola.ccc.mainplm
com.motorola.ccc.notification
com.motorola.demo
com.motorola.dynamicvolume
com.motorola.easyprefix
com.motorola.fmplayer
com.motorola.freeform
com.motorola.gamemode
com.motorola.gesture
com.motorola.help
com.motorola.launcher3
com.motorola.launcherconfig.overlay.*
com.motorola.moto
com.motorola.motosignature.app
com.motorola.motosignature2.app
com.motorola.personalize
com.motorola.screenshoteditor
com.motorola.spaces
com.motorola.timeweatherwidget
com.motorola.wifi.motowifimetrics
com.pocketgeek.xfinity.android
com.spectrum.cm.headless
com.tmobile.*
com.tracfone.*
com.verizon.*
com.xfinitymobile.cometcarrierservice
```

---

## What You CANNOT Disable

Some packages are protected by the system:

```
com.motorola.paks              # "Cannot disable a protected package"
com.motorola.ccc.devicemanagement
com.amazon.appmanager          # "non-disable"
com.google.android.inputmethod.latin  # Keyboard (disable = no typing)
```

For these, you'd need to use Magisk modules or modify /system directly.

---

## Recommended Replacement Apps

Since we disabled many system apps, here are lightweight alternatives:

| Function | Disabled | Replacement |
|----------|----------|-------------|
| Launcher | com.motorola.launcher3 | Lawnchair, KISS Launcher, or Termux:Widget |
| Keyboard | (keep Gboard or) | OpenBoard, AnySoftKeyboard |
| Camera | com.motorola.camera3 | Open Camera |
| Browser | com.android.chrome | Firefox, Bromite |
| File Manager | com.google.android.apps.nbu.files | Material Files, Termux |

---

## Going Further: Disable SystemUI

**WARNING: This makes the phone nearly unusable for normal Android use**

If you want maximum RAM and only plan to use the phone as a headless AI server:

```bash
# This disables the entire Android UI
adb shell "su -c 'pm disable-user --user 0 com.android.systemui'"

# To re-enable (you'll need to do this via ADB):
adb shell "su -c 'pm enable com.android.systemui'"
```

With SystemUI disabled:
- No status bar, no notifications, no navigation
- Phone becomes SSH/ADB-only device
- Saves ~250MB RAM
- Screen stays black (can still use via ADB)

---

## Reverting Changes

### Re-enable all disabled packages
```bash
for pkg in $(adb shell "pm list packages -d" | sed 's/package://'); do
    adb shell "su -c 'pm enable $pkg'"
done
```

### Factory reset (nuclear option)
```bash
adb reboot recovery
# Select "Wipe data/factory reset"
```

---

## Memory Optimization Tips

### 1. Reduce zygote preload
Edit `/system/build.prop` (requires remounting /system as rw):
```
dalvik.vm.dex2oat-threads=2
dalvik.vm.image-dex2oat-threads=2
```

### 2. Disable animations
```bash
adb shell "settings put global window_animation_scale 0"
adb shell "settings put global transition_animation_scale 0"
adb shell "settings put global animator_duration_scale 0"
```

### 3. Force GPU rendering
```bash
adb shell "settings put global force_gpu_rendering 1"
```

### 4. Limit background processes
```bash
adb shell "settings put global background_process_limit 2"
```

---

## Verifying Results

```bash
# Check free memory
adb shell "su -c 'free -m'"

# Count running processes
adb shell "su -c 'ps -A | wc -l'"

# Top memory consumers
adb shell "su -c 'ps -A -o rss,name --sort=-rss | head -20'"

# Disabled packages count
adb shell "pm list packages -d | wc -l"
```

---

## For TriX/AI Server Use

After debloating, the phone has enough free RAM to run:
- LFM2-1.2B at 3840 context (~17 tok/s)
- With room for Qdrant vector DB
- And the Termux boot script

See `setup_termux.sh` for the auto-start configuration.

---

*Last updated: February 4, 2026*
*Device: Moto G Power 5G 2023 (XT2311-4), Android 14*
