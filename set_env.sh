#!/usr/bin/env bash
# Source this file before running python3 run_test.py.

export CANN_HOME="${CANN_HOME:-/home/lenovo/.codex/memories/cann-9.0.0/cann-9.0.0}"
export ARCH="${ARCH:-x86_64-linux}"

# Compiler/runtime SoC name.
export SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
export NPU_TYPE="${NPU_TYPE:-$SOC_VERSION}"
export CORE_ARCH="${CORE_ARCH:-dav-c310-vec}"
if [ -z "${CORE_SIM_DIR:-}" ]; then
  case "$SOC_VERSION:$CORE_ARCH" in
    *Ascend910_9691*|*dav_9201*|*dav-310r6*)
      export CORE_SIM_DIR="dav_9201"
      ;;
    *)
      export CORE_SIM_DIR="dav_3510"
      ;;
  esac
else
  export CORE_SIM_DIR
fi

# Optional: point this to another tools/simulator directory when the compiler
# package and the full-dump simulator package are split.
#
# Example:
#   export FULL_SIMULATOR_HOME=/home/user/full_dump_package/tools/simulator
export FULL_SIMULATOR_HOME="${FULL_SIMULATOR_HOME:-}"

export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
export ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}"

if [ -f "$CANN_HOME/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$CANN_HOME/set_env.sh"
fi

export ASCEND_HOME_PATH="$CANN_HOME"
export ASCEND_CANN_PACKAGE_PATH="$CANN_HOME"
export ASCEND_TOOLKIT_HOME="$CANN_HOME"
export ASCEND_OPP_PATH="$CANN_HOME/opp"
export PYTHONPATH="$CANN_HOME/python/site-packages:${PYTHONPATH:-}"

prepend_path() {
  local d="$1"
  [ -d "$d" ] && export PATH="$d:${PATH:-}"
}

prepend_ld_path() {
  local d="$1"
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
}

prepend_path "$CANN_HOME/$ARCH/bin"
prepend_path "$CANN_HOME/$ARCH/simulator/bin"
prepend_path "$CANN_HOME/bin"
prepend_path "$CANN_HOME/compiler/bin"

prepend_ld_path "$CANN_HOME/tools/simulator/$SOC_VERSION/camodel"
prepend_ld_path "$CANN_HOME/tools/simulator/$SOC_VERSION/lib"
prepend_ld_path "$CANN_HOME/$ARCH/simulator/$CORE_SIM_DIR/camodel"
prepend_ld_path "$CANN_HOME/$ARCH/simulator/$CORE_SIM_DIR/lib"
prepend_ld_path "$CANN_HOME/$ARCH/simulator/$SOC_VERSION/lib"
prepend_ld_path "$CANN_HOME/$ARCH/devlib/linux/aarch64"
prepend_ld_path "$CANN_HOME/$ARCH/devlib/linux/x86_64"
prepend_ld_path "$CANN_HOME/$ARCH/devlib/device"
prepend_ld_path "$CANN_HOME/$ARCH/devlib"
prepend_ld_path "$CANN_HOME/$ARCH/lib64/device/lib64"
prepend_ld_path "$CANN_HOME/$ARCH/lib64"
prepend_ld_path "$CANN_HOME/runtime/lib64"
prepend_ld_path "$CANN_HOME/fwkacllib/lib64"
prepend_ld_path "$CANN_HOME/lib64"

if [ -n "$FULL_SIMULATOR_HOME" ]; then
  prepend_ld_path "$FULL_SIMULATOR_HOME/$CORE_SIM_DIR/lib"
  prepend_ld_path "$FULL_SIMULATOR_HOME/$CORE_SIM_DIR/camodel"
  prepend_ld_path "$FULL_SIMULATOR_HOME/$SOC_VERSION/lib"
  prepend_ld_path "$FULL_SIMULATOR_HOME/$SOC_VERSION/camodel"
fi

unset LD_PRELOAD

echo "[INFO] CANN_HOME=$CANN_HOME"
echo "[INFO] ARCH=$ARCH"
echo "[INFO] SOC_VERSION=$SOC_VERSION"
echo "[INFO] CORE_ARCH=$CORE_ARCH"
echo "[INFO] CORE_SIM_DIR=$CORE_SIM_DIR"
if [ -n "$FULL_SIMULATOR_HOME" ]; then
  echo "[INFO] FULL_SIMULATOR_HOME=$FULL_SIMULATOR_HOME"
fi
