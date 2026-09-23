#!/usr/bin/env bash
# Source this from any directory after setting CANN_HOME.
# CANN_HOME: compiler/toolkit package, containing set_env.sh.
# FULL_SIMULATOR_HOME: optional second package's tools/simulator directory,
# containing dav_9201/{lib,camodel} or Ascend910_9691/{lib,camodel}.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source benchmarks/a6/set_env.sh" >&2
    exit 1
fi
if [[ -n "${FULL_SIMULATOR_HOME:-}" && ! -d "$FULL_SIMULATOR_HOME/dav_9201/lib" &&
      ! -d "$FULL_SIMULATOR_HOME/dav_9201/camodel" &&
      ! -d "$FULL_SIMULATOR_HOME/Ascend910_9691/lib" &&
      ! -d "$FULL_SIMULATOR_HOME/Ascend910_9691/camodel" ]]; then
    echo "FULL_SIMULATOR_HOME must contain dav_9201 or Ascend910_9691 simulator directories." >&2
    return 1
fi
if [[ -z "${CANN_HOME:-}" || ! -f "$CANN_HOME/set_env.sh" ]]; then
    echo "Set CANN_HOME to your toolkit installation (containing set_env.sh)." >&2
    return 1
fi
export SOC_VERSION=Ascend910_9691
export NPU_TYPE=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201
export ARCH="${ARCH:-$(uname -m)-linux}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/set_env.sh"
