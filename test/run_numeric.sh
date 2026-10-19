#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/db4ai-numeric.XXXXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
CC=${CC:-cc}
SIMD=${WITH_SIMD:-1}
SIMD_FLAGS=''
if [ "$SIMD" = 1 ] && printf 'int main(void){return 0;}\n' | "$CC" -x c -fopenmp-simd -c -o "$TMP/probe.o" - 2>/dev/null; then
    SIMD_FLAGS='-fopenmp-simd'
else
    SIMD=0
fi
"$CC" -std=c11 -O3 -ffp-contract=off $SIMD_FLAGS -DDB4AI_WITH_SIMD="$SIMD" \
    -I"$ROOT/src" "$ROOT/src/numeric.c" "$ROOT/test/numeric_simd.c" -lm -o "$TMP/test"
"$TMP/test"
