#!/usr/bin/env bash
# Program the DE10-Standard FPGA only after detecting and validating its JTAG chain.
set -euo pipefail

quartus_bin=/home/raone/intelFPGA/25.1lite/quartus/bin
jtagconfig="$quartus_bin/jtagconfig"
quartus_pgm="$quartus_bin/quartus_pgm"
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sof="$project_dir/cnn_full_small.sof"

if [[ ! -x "$jtagconfig" || ! -x "$quartus_pgm" ]]; then
    echo "ERROR: Quartus programmer tools are unavailable under $quartus_bin." >&2
    exit 1
fi
if [[ ! -f "$sof" ]]; then
    echo "ERROR: SOF not found: $sof" >&2
    exit 1
fi

echo "Detecting JTAG hardware..."
cables=$($jtagconfig)
printf '%s\n' "$cables"

cable=$(printf '%s\n' "$cables" |
    sed -nE 's/^[[:space:]]*[0-9]+\)[[:space:]]*(DE-SoC[[:space:]]+\[[^]]+\]).*/\1/p' |
    head -n 1)

if [[ -z "$cable" ]]; then
    echo "ERROR: No DE-SoC JTAG cable detected; FPGA was not programmed." >&2
    exit 1
fi

echo "Selected cable: $cable"
echo "Reading JTAG chain..."
chain=$($jtagconfig -c "$cable")
printf '%s\n' "$chain"

if ! printf '%s\n' "$chain" | grep -Eq '02D020DD[[:space:]]+5CSEBA6'; then
    echo "ERROR: Expected FPGA device 02D020DD / 5CSEBA6 at JTAG index 2 was not found; FPGA was not programmed." >&2
    exit 1
fi

echo "Programming $sof to JTAG device @2..."
program_output=$($quartus_pgm -c "$cable" -m jtag -o "p;$sof@2" 2>&1) || {
    printf '%s\n' "$program_output" >&2
    echo "ERROR: Programming failed." >&2
    exit 1
}
printf '%s\n' "$program_output"

if printf '%s\n' "$program_output" | grep -Eq '(^|[[:space:]])(Error|Warning)[[:space:]]*\('; then
    echo "ERROR: Programmer reported an error or warning; configuration is not accepted as clean." >&2
    exit 1
fi

echo "Configuration succeeded: 0 errors, 0 warnings."
