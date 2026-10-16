#!/bin/sh
# Runs iiama2.dol's bench mode headless in Dolphin and prints the results.
#
#   tools/dolphin_bench.sh sd/llama          # folder with model.bin + tokenizer.bin
#   tools/dolphin_bench.sh sd/llama 32       # tokens per engine (default 64)
#   TEMP=0.8 tools/dolphin_bench.sh sd/llama # then every engine again, sampling at that temperature
#   PROBE=1 tools/dolphin_bench.sh sd/llama  # GPU and DSP probes instead (gx_probe.c, dsp_probe.c);
#                                            # keeps probe.bin and probe_out.txt
#
# Builds a FAT SD card image with mtools (brew install mtools), and uses a
# throwaway Dolphin user folder so your own settings are untouched. Runs the
# software renderer, whose TEV math is what tools/tev_sim copies. The probes also
# run the DSP in low-level emulation (interpreter, with Dolphin's own boot ROM from
# Sys/GC), since high-level emulation only knows Nintendo's microcode.
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
[ -n "${TEMP:-}" ] && printf 'temp %s\n' "$TEMP" >> "$WORK/$MODE_FILE"
DSP_CORE= DSP_JIT= DSP_LOG=
if [ -n "${PROBE:-}" ]; then
    DSP_CORE="DSPHLE = False"
    DSP_JIT="EnableJIT = False"
    DSP_LOG="DSPLLE = True"
fi
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
$DSP_CORE
[General]
WiiSDCardPath = $SD
[Interface]
UsePanicHandlers = False
[Analytics]
PermissionAsked = True
Enabled = False
[DSP]
Backend = No Audio Output
$DSP_JIT
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
$DSP_LOG
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
[ -n "${PROBE:-}" ] && mcopy -i "$SD" ::/llama/probe.bin ::/llama/probe_out.txt "$WORK/"
echo "(Dolphin files: $WORK)"
