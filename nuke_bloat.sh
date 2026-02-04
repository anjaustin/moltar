#!/bin/bash
# NUCLEAR DEBLOAT - Remove EVERYTHING that's not essential
# This makes the phone yours. No Google. No Comcast. No Motorola bullshit.
# 
# WARNINGS:
# - No Play Store, no Google apps
# - No carrier features (VoLTE might break)
# - Install F-Droid/Aurora Store first if you want apps
# - Install a keyboard (OpenBoard) first or keep Gboard
#
# Run with: bash nuke_bloat.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${RED}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           NUCLEAR DEBLOAT - NO MERCY MODE                 ║"
echo "║                                                           ║"
echo "║  This will disable EVERYTHING non-essential.              ║"
echo "║  Your phone. Your rules. Fuck Comcast.                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check root
if ! adb shell "su -c 'id'" 2>/dev/null | grep -q "uid=0"; then
    echo -e "${RED}ERROR: Root access required${NC}"
    exit 1
fi

disable_pkg() {
    pkg=$1
    result=$(adb shell "su -c 'pm disable-user --user 0 $pkg'" 2>&1)
    if echo "$result" | grep -q "disabled"; then
        echo -e "${GREEN}[NUKED]${NC} $pkg"
        return 0
    elif echo "$result" | grep -q "not found\|Unknown"; then
        return 1  # Silent for not found
    else
        echo -e "${YELLOW}[PROTECTED]${NC} $pkg"
        return 1
    fi
}

# Count successes
count=0

echo -e "\n${CYAN}=== PHASE 1: GOOGLE - ALL OF IT ===${NC}"
google_pkgs=(
    "com.google.android.gms"
    "com.google.android.gsf"
    "com.google.android.apps.googleassistant"
    "com.google.android.apps.docs.editors.sheets"
    "com.google.android.apps.safetyhub"
    "com.google.android.apps.scone"
    "com.google.android.apps.subscriptions.red"
    "com.google.android.apps.wallpaper"
    "com.google.android.apps.youtube.music.setupwizard"
    "com.google.android.apps.cbrsnetworkmonitor"
    "com.google.android.as.oss"
    "com.google.android.carrier"
    "com.google.android.cellbroadcastreceiver"
    "com.google.android.cellbroadcastservice"
    "com.google.android.configupdater"
    "com.google.android.euicc"
    "com.google.android.ext.services"
    "com.google.android.federatedcompute"
    "com.google.android.gms.location.history"
    "com.google.android.health.connect.backuprestore"
    "com.google.android.healthconnect.controller"
    "com.google.android.hotspot2.osulogin"
    "com.google.android.onetimeinitializer"
    "com.google.android.partnersetup"
    "com.google.android.setupwizard"
    "com.google.android.tag"
    "com.google.android.tts"
    "com.google.android.wfcactivation"
    "com.google.android.wifi.dialog"
    "com.google.mainline.adservices"
    "com.google.mainline.telemetry"
    "com.google.ambient.streaming"
    "com.android.hotwordenrollment.okgoogle"
    "com.android.hotwordenrollment.xgoogle"
)

for pkg in "${google_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 2: COMCAST/XFINITY - BURN IT ===${NC}"
comcast_pkgs=(
    "com.motorola.comcast.settings.extensions"
    "com.motorola.comcastext"
    "com.motorola.settings.overlay.comcast"
    "com.motorola.setup.overlay.comcast"
    "com.motorola.android.overlay.comcast.cbrs"
    "com.motorola.android.launcher.overlay.comcast"
    "com.android.providers.settings.overlay.comcast"
    "com.xfinitymobile.cometcarrierservice"
    "com.pocketgeek.xfinity.android"
)

for pkg in "${comcast_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 3: MOTOROLA BLOAT ===${NC}"
moto_pkgs=(
    "com.motorola.motocare"
    "com.motorola.ccc"
    "com.motorola.ccc.ota"
    "com.motorola.ccc.devicemanagement"
    "com.motorola.ccc.mainplm"
    "com.motorola.ccc.notification"
    "com.motorola.dolby.dolbyui"
    "com.motorola.omadm.service"
    "com.motorola.omadm.vzw"
    "com.motorola.mobiledesktop.core"
    "com.motorola.sarcontrol"
    "com.motorola.mototour"
    "com.motorola.discovery"
    "com.motorola.genie"
    "com.motorola.appforecast"
    "com.motorola.bug2go"
    "com.motorola.help.extlog"
    "com.motorola.hiddenmenuapp"
    "com.motorola.lifetimedata"
    "com.motorola.livewallpaper3"
    "com.motorola.motocit"
    "com.motorola.nfwlocationattribution"
    "com.motorola.revoker.services"
    "com.motorola.securevault"
    "com.motorola.securityhub"
    "com.motorola.securityhubext"
    "com.motorola.slpc_sys"
    "com.motorola.smart5g"
    "com.motorola.systemserver"
    "com.motorola.systemui.desk"
    "com.motorola.thermalservice"
    "com.motorola.appdirectedsmsproxy"
    "com.motorola.callredirectionservice"
    "com.motorola.camera3.content.ai"
    "com.motorola.contacts.preloadcontacts"
    "com.motorola.enterprise.adapter.service"
    "com.motorola.enterprise.service"
    "com.motorola.entitlement"
    "com.motorola.faceunlock"
    "com.motorola.imagertuning_u"
    "com.motorola.bach.modemstats"
    "com.motorola.attvowifi"
    "com.motorola.vzw.pco.extensions.pcoreceiver"
    "com.motorola.nfc"
    "com.motorola.paks"
    "com.motorola.paks.notification"
    "com.motorola.android.nativedropboxagent"
    "com.motorola.android.fmradio"
    "com.motorola.android.providers.chromehomepage"
)

for pkg in "${moto_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 4: CARRIER GARBAGE ===${NC}"
carrier_pkgs=(
    "com.verizon.loginengine.unbranded"
    "com.verizon.remoteSimlock"
    "com.tmobile.echolocate"
    "com.tmobile.echolocate.system"
    "com.tmobile.pr.adapt"
    "com.aura.oobe.motorola"
    "com.aura.services.tmobile"
    "com.ironsrc.aura.tmo"
    "com.ironsrc.aura.appmanager.tmo"
    "com.metro.minus1"
    "com.metropcs.metrozone"
    "com.tracfone.generic.mysites"
    "com.tracfone.preload.accountservices"
    "com.swishme.tracfone"
    "com.spectrum.cm.headless"
    "com.nuance.nmc.sihome.metropcs"
)

for pkg in "${carrier_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 5: FACEBOOK/AMAZON ===${NC}"
fb_pkgs=(
    "com.facebook.katana"
    "com.facebook.orca"
    "com.facebook.services"
    "com.facebook.system"
    "com.facebook.appmanager"
    "com.amazon.appmanager"
)

for pkg in "${fb_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 6: ANDROID BLOAT ===${NC}"
android_pkgs=(
    "com.android.chrome"
    "com.android.vending"
    "com.android.stk"
    "com.android.bips"
    "com.android.printspooler"
    "com.android.bookmarkprovider"
    "com.android.dreams.basic"
    "com.android.egg"
    "com.android.wallpaper.livepicker"
    "com.android.soundpicker"
)

for pkg in "${android_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 7: MORE GOOGLE OVERLAYS ===${NC}"
overlay_pkgs=(
    "com.google.android.overlay.gmsconfig.asi"
    "com.google.android.overlay.gmsconfig.common"
    "com.google.android.overlay.gmsconfig.comms"
    "com.google.android.overlay.gmsconfig.geotz"
    "com.google.android.overlay.gmsconfig.gsa"
    "com.google.android.overlay.gmsconfig.personalsafety"
    "com.google.android.overlay.gmsconfig.photos"
)

for pkg in "${overlay_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo -e "\n${CYAN}=== PHASE 8: MEDIATEK BLOAT ===${NC}"
mtk_pkgs=(
    "com.mediatek.presence"
    "com.mediatek.carrierexpress"
)

for pkg in "${mtk_pkgs[@]}"; do
    if disable_pkg "$pkg"; then ((count++)); fi
done

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              NUCLEAR DEBLOAT COMPLETE                     ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║  Packages nuked: $count                                       ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║  Reboot to apply: adb reboot                              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Your phone is now yours."
echo ""
echo "Recommended next steps:"
echo "  1. adb reboot"
echo "  2. Install F-Droid for apps"
echo "  3. Install a minimal launcher (KISS, Lawnchair)"
echo "  4. Install OpenBoard keyboard if Gboard was disabled"
