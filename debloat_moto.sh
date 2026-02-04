#!/bin/bash
# Aggressive debloat for Moto G Power 5G 2023
# Run with: bash debloat_moto.sh
# To restore: pm enable <package> or factory reset

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Moto G Power 5G 2023 Debloat ===${NC}"
echo "This will disable bloatware to free up RAM"
echo "Packages are DISABLED, not uninstalled (recoverable)"
echo ""

# Check root
if ! adb shell "su -c 'id'" 2>/dev/null | grep -q "uid=0"; then
    echo -e "${RED}ERROR: Root access required${NC}"
    exit 1
fi

disable_pkg() {
    pkg=$1
    result=$(adb shell "su -c 'pm disable-user --user 0 $pkg'" 2>&1)
    if echo "$result" | grep -q "disabled"; then
        echo -e "${GREEN}[DISABLED]${NC} $pkg"
    elif echo "$result" | grep -q "not found"; then
        echo -e "${YELLOW}[NOT FOUND]${NC} $pkg"
    else
        echo -e "${RED}[FAILED]${NC} $pkg - $result"
    fi
}

echo -e "\n${YELLOW}--- Google Bloat (saves ~400MB RAM) ---${NC}"
# Google Play Services - THE BIG ONE
# WARNING: This breaks Play Store, Google apps, some app logins
disable_pkg "com.google.android.gms"
disable_pkg "com.google.android.gsf"

# Google apps
disable_pkg "com.google.android.youtube"
disable_pkg "com.google.android.apps.youtube.music"
disable_pkg "com.google.android.apps.maps"
disable_pkg "com.google.android.apps.photos"
disable_pkg "com.google.android.apps.docs"
disable_pkg "com.google.android.apps.tachyon"  # Duo
disable_pkg "com.google.android.videos"
disable_pkg "com.google.android.music"
disable_pkg "com.google.android.apps.magazines"
disable_pkg "com.google.android.apps.books"
disable_pkg "com.google.android.keep"
disable_pkg "com.google.android.calendar"
disable_pkg "com.google.android.gm"  # Gmail
disable_pkg "com.google.android.googlequicksearchbox"
disable_pkg "com.google.android.inputmethod.latin"  # Gboard
disable_pkg "com.google.android.talk"  # Hangouts
disable_pkg "com.google.android.apps.messaging"
disable_pkg "com.google.android.apps.wellbeing"
disable_pkg "com.google.android.apps.turbo"  # Device Health
disable_pkg "com.google.android.feedback"
disable_pkg "com.google.android.printservice.recommendation"
disable_pkg "com.google.android.apps.nbu.files"  # Files
disable_pkg "com.google.android.dialer"
disable_pkg "com.google.android.contacts"
disable_pkg "com.google.android.deskclock"
disable_pkg "com.google.android.calculator"
disable_pkg "com.google.android.apps.walletnfcrel"  # Google Pay
disable_pkg "com.google.android.projection.gearhead"  # Android Auto
disable_pkg "com.google.ar.core"  # AR
disable_pkg "com.google.android.marvin.talkback"  # Accessibility (keep if needed)
disable_pkg "com.google.android.accessibility.switchaccess"
disable_pkg "com.google.android.apps.restore"
disable_pkg "com.google.android.apps.work.oobconfig"
disable_pkg "com.google.android.adservices.api"
disable_pkg "com.google.android.ondevicepersonalization.services"
disable_pkg "com.android.vending"  # Play Store itself

echo -e "\n${YELLOW}--- Motorola Bloat ---${NC}"
disable_pkg "com.motorola.launcher3"  # Use a lighter launcher
disable_pkg "com.motorola.timeweatherwidget"
disable_pkg "com.motorola.ccc.notification"
disable_pkg "com.motorola.ccc.mainplm"
disable_pkg "com.motorola.ccc.devicemanagement"
disable_pkg "com.motorola.paks"
disable_pkg "com.motorola.help"
disable_pkg "com.motorola.demo"
disable_pkg "com.motorola.brapps"
disable_pkg "com.motorola.gametime"
disable_pkg "com.motorola.myfeedsvideo"
disable_pkg "com.motorola.actions"
disable_pkg "com.motorola.gesture"
disable_pkg "com.motorola.screenshoteditor"
disable_pkg "com.motorola.camera3"  # Can use open camera
disable_pkg "com.motorola.gallery"
disable_pkg "com.motorola.fmplayer"
disable_pkg "com.motorola.audiorecorder"
disable_pkg "com.motorola.dolby"
disable_pkg "com.motorola.aiservices"
disable_pkg "com.motorola.dynamicvolume"
disable_pkg "com.motorola.wifi.motowifimetrics"
disable_pkg "com.motorola.android.provisioning"

echo -e "\n${YELLOW}--- Third-party Bloat ---${NC}"
disable_pkg "com.inmobi.weather"  # That 154MB weather app!
disable_pkg "com.facebook.katana"
disable_pkg "com.facebook.orca"
disable_pkg "com.facebook.services"
disable_pkg "com.facebook.system"
disable_pkg "com.facebook.appmanager"
disable_pkg "com.amazon.appmanager"
disable_pkg "com.amazon.mp3"
disable_pkg "com.amazon.mShop.android.shopping"
disable_pkg "com.linkedin.android"
disable_pkg "com.spotify.music"
disable_pkg "com.netflix.mediaclient"
disable_pkg "com.netflix.partner.activation"

echo -e "\n${YELLOW}--- Android Bloat (safe to disable) ---${NC}"
disable_pkg "com.android.chrome"  # Use Firefox/other
disable_pkg "com.android.email"
disable_pkg "com.android.calendar"
disable_pkg "com.android.contacts"
disable_pkg "com.android.deskclock"
disable_pkg "com.android.calculator2"
disable_pkg "com.android.gallery3d"
disable_pkg "com.android.music"
disable_pkg "com.android.musicfx"
disable_pkg "com.android.soundrecorder"
disable_pkg "com.android.dreams.basic"
disable_pkg "com.android.dreams.phototable"
disable_pkg "com.android.printspooler"
disable_pkg "com.android.bips"  # Print service
disable_pkg "com.android.bookmarkprovider"
disable_pkg "com.android.stk"  # SIM toolkit
disable_pkg "com.android.wallpaper.livepicker"

echo -e "\n${YELLOW}=== Debloat Complete ===${NC}"
echo ""
echo "Reboot for changes to take full effect:"
echo "  adb reboot"
echo ""
echo "To re-enable a package:"
echo "  adb shell su -c 'pm enable <package>'"
echo ""
echo "To see disabled packages:"
echo "  adb shell pm list packages -d"
