# microbenchmark-cce-direct

This repository compiles one CCE kernel and runs it directly through the CANN camodel runtime. It does not call `msprof op simulator` or `cannsim`.

The default kernel is `op_kernel/kernel.cce`, a SIMD VF `A = B + C` fp32 example with GM -> UB -> vector compute -> GM.

## 1. Environment

On a new machine, set `CANN_HOME` to the root directory that contains `set_env.sh`.

```bash
cd /path/to/microbenchmark-cce-direct
export CANN_HOME=/absolute/path/to/cann-9.0.0
export ARCH=x86_64-linux        # use aarch64-linux on an aarch64 host/package
export SOC_VERSION=Ascend950PR_9599
export CORE_ARCH=dav-c310-vec
export CORE_SIM_DIR=dav_3510
source ./set_env.sh
```

This local copy defaults to:

```bash
CANN_HOME=/home/lenovo/.codex/memories/cann-9.0.0/cann-9.0.0
ARCH=x86_64-linux
SOC_VERSION=Ascend950PR_9599
CORE_ARCH=dav-c310-vec
CORE_SIM_DIR=dav_3510
```

For A6/Ascend910_9691, use the A6 simulator directory:

```bash
export SOC_VERSION=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201
source ./set_env.sh
python3 run_test.py
```

The verified A6 NDDMA/scatter data-movement flow is documented in
[`docs/A6_CAMODEL_GUIDE.md`](docs/A6_CAMODEL_GUIDE.md).

If `CORE_SIM_DIR` is not set, `set_env.sh` and `run_test.py` infer it from `SOC_VERSION`/`CORE_ARCH`:

```text
Ascend910_9691, dav-920r1 or dav-310r6 -> dav_9201
otherwise                            -> dav_3510
```

If the compiler/runtime package and the full-dump simulator package are separate, keep `CANN_HOME` pointing to the compiler/runtime package and set `FULL_SIMULATOR_HOME` to the standalone `tools/simulator` directory:

```bash
export CANN_HOME=/absolute/path/to/closed/cann-9.1.0
export FULL_SIMULATOR_HOME=/absolute/path/to/full_dump_package/tools/simulator
source ./set_env.sh
python3 run_test.py
```

When `FULL_SIMULATOR_HOME` is set, its simulator libraries are placed before the simulator libraries under `CANN_HOME`:

```text
$FULL_SIMULATOR_HOME/$SOC_VERSION/camodel
$FULL_SIMULATOR_HOME/$SOC_VERSION/lib
$FULL_SIMULATOR_HOME/$CORE_SIM_DIR/camodel
$FULL_SIMULATOR_HOME/$CORE_SIM_DIR/lib
```

`run_test.py` also searches `FULL_SIMULATOR_HOME` first when copying `1982_cloud_config.toml` into the runner working directory.

To confirm which simulator is actually loaded, inspect `result/<timestamp>/debug.log`. Each run records:

```text
[INFO] runtime FULL_SIMULATOR_HOME=...
[INFO] runtime LD_LIBRARY_PATH first entries:
[INFO] native runner linked simulator libs:
```

The `ldd` output should resolve simulator-related libraries to `FULL_SIMULATOR_HOME` when an external simulator is intended.

## 2. Run

```bash
python3 run_test.py
```

Use another CCE file:

```bash
python3 run_test.py --kernel op_kernel/your_kernel.cce
```

Only compile without running:

```bash
python3 run_test.py --compile-only
```

Write results to a fixed directory:

```bash
python3 run_test.py --output result/my_case
```

## 3. Kernel Contract

The easiest mode is generic tensor mode. Put these comments in the CCE file:

```cpp
// MB_KERNEL add_simd
// MB_TENSOR output A fp32 256
// MB_TENSOR input B fp32 256
// MB_TENSOR input C fp32 256
// MB_GOLDEN A B + C
```

Meaning:

- `MB_KERNEL`: exported kernel symbol passed to the runner.
- `MB_TENSOR`: one kernel argument, in the same order as the kernel function arguments.
- `MB_GOLDEN`: optional elementwise precision check expression.

Supported tensor directions are `input`, `output`, and `inout`. Supported shapes can be a single element count like `256`, a product like `8x1024`, or `TOTAL_COUNT`.

## 4. Outputs

Each run writes to `result/<timestamp>/` unless `--output` is specified.

```text
result/<timestamp>/build/<kernel>_mix_aiv.o
result/<timestamp>/build/<kernel>_mix.o
result/<timestamp>/build/native_cce_runner
result/<timestamp>/build/kernel_tensors.tsv
result/<timestamp>/run/input_*.bin
result/<timestamp>/run/output_*.bin
result/<timestamp>/run/golden_check.log
result/<timestamp>/run/log_ca/*.dump
result/<timestamp>/debug.log
result/<timestamp>/model.log
```

`debug.log` contains framework diagnostics such as the selected CANN paths,
compiler output, runtime environment, and resolved simulator libraries.
`model.log` contains the unmodified stdout/stderr produced while running the
native runner and CAmodel. Model output is not streamed to the terminal; if a
run fails, only its last 30 lines are printed there.

Important dump files are under:

```text
result/<timestamp>/run/log_ca/
```

For example:

```text
core0.veccore0.instr_popped_log.dump
core0.veccore0.lsu.dump
core0.veccore0.rvec.exu0.dump
```

## 5. Common Options

```bash
python3 run_test.py \
  --kernel op_kernel/kernel.cce \
  --block-dim 1 \
  --local-memory-size 0 \
  --device-id 0 \
  --atol 1e-5 \
  --rtol 1e-5
```

`--local-memory-size` is the dynamic UB/local memory size passed to the runtime. `0` keeps the runtime default.

## 6. Troubleshooting

If `acl/acl.h` is missing, check that `CANN_HOME` points to the CANN root and that `$CANN_HOME/$ARCH/include` exists.

If `libascend_hal.so` or `libruntime_camodel.so` is missing, check `ARCH`, `SOC_VERSION`, and `LD_LIBRARY_PATH` after `source ./set_env.sh`.

If `log_ca` exists but detailed dump content is incomplete, check the simulator config for this SoC. For Ascend950PR_9599, `Flush_level` often needs to be set to `2`.

If `--soc-version is invalid`, confirm the directory exists:

```bash
ls "$CANN_HOME/tools/simulator/$SOC_VERSION" "$CANN_HOME/$ARCH/simulator"
```
