#!/bin/sh
# Runs iiama2.dol's bench mode headless in Dolphin and prints the results.
#
#   tools/dolphin_bench.sh sd/llama          # folder with model.bin + tokenizer.bin
#   tools/dolphin_bench.sh sd/llama 32       # tokens per engine (default 64)
#   PROBE=1 tools/dolphin_bench.sh sd/llama  # GPU probes instead (gx_probe.c); keeps probe.bin
#
# Builds a FAT SD card image with mtools (brew install mtools), and uses a
# throwaway Dolphin user folder so your own settings are untouched. Runs the
# software renderer, whose TEV math is what tools/tev_sim copies.
#
# Speed numbers from Dolphin mean nothing; compare the text and the selftest
# hashes against tools/tev_sim/check_dolphin.py instead.
set -eu
LLAMA_DIR=${1:?usage: tools/dolphin_bench.sh <folder with model.bin + tokenizer.bin> [steps]}
STEPS=${2:-64}
DOL=${DOL:-$(pwd)/iiama2.dol}
DOLPHIN=${DOLPHIN:-/Applications/Dolphin.app/Contents/MacOS/Dolphin}
TIMEOUT=${TIMEOUT:-1800}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/iiama2-dolphin.XXXXXX")
mkdir -p "$WORK/user/Config"

# SD card: 128 MB FAT image with /llama/{model.bin, tokenizer.bin, bench.txt}
SD="$WORK/sd.raw"
MODE_FILE=bench.txt
[ -n "${PROBE:-}" ] && MODE_FILE=probe.txt
printf 'poweroff\nsteps %s\n' "$STEPS" > "$WORK/$MODE_FILE"
dd if=/dev/zero of="$SD" bs=1048576 count=128 2>/dev/null
mformat -i "$SD" -T 262144 -h 64 -s 32 -F ::
mmd -i "$SD" ::/llama
mcopy -i "$SD" "$LLAMA_DIR/model.bin" "$LLAMA_DIR/tokenizer.bin" "$WORK/$MODE_FILE" ::/llama/

cat > "$WORK/user/Config/Dolphin.ini" <<EOF
[Core]
WiiSDCard = True
WiiSDCardAllowWrites = True
WiiSDCardEnableFolderSync = False
GFXBackend = Software Renderer
EmulationSpeed = 0.00000000
[General]
WiiSDCardPath = $SD
[Interface]
UsePanicHandlers = False
[Analytics]
PermissionAsked = True
Enabled = False
[DSP]
Backend = No Audio Output
EOF
cat > "$WORK/user/Config/Logger.ini" <<EOF
[Options]
Verbosity = 4
WriteToFile = True
WriteToConsole = False
WriteToWindow = False
[Logs]
OSREPORT = True
OSREPORT_HLE = True
EOF

"$DOLPHIN" -u "$WORK/user" -b -e "$DOL" > "$WORK/stdout.txt" 2>&1 &
PID=$!

LOG="$WORK/user/Logs/dolphin.log"
waited=0
while kill -0 "$PID" 2>/dev/null && [ "$waited" -lt "$TIMEOUT" ]; do
    if [ -f "$LOG" ] && grep -q "OSREPORT.*done" "$LOG"; then break; fi
    sleep 2
    waited=$((waited + 2))
done
sleep 2
kill "$PID" 2>/dev/null || true

if [ -f "$LOG" ] && grep -q OSREPORT "$LOG"; then
    sed -n 's/.*OSREPORT[A-Z_]*\]: //p' "$LOG"
else
    echo "no output from the app in Dolphin's log; Dolphin said:" >&2
    cat "$WORK/stdout.txt" >&2
    [ -f "$LOG" ] && cat "$LOG" >&2
    exit 1
fi
[ -n "${PROBE:-}" ] && mcopy -i "$SD" ::/llama/probe.bin "$WORK/probe.bin"
echo "(Dolphin files: $WORK)"
