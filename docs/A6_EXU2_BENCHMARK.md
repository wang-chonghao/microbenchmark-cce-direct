# A6 EXU2 supplementary benchmarks

## Scope and evidence

This supplement compares the working-tree version of
`VfSimulator/.worktrees/MultiSoC-membar-refactor/configs/socs/DV121/isa.json`
with the original `op_kernel/a6_instruction/manifest.json`.
The snapshot is `benchmarks/a6/exu2_inventory.json`; it records the source SHA256
and only the instruction/form inventory, not the model's latency values.

The worktree has 172 EXU2-eligible instruction/form entries. The original suite
contains probes for 20 of these; 152 entries lacked probes. These are source
coverage counts, NOT proof that previous remote runs passed or measured EXU2.
Some routing entries in DV121 are inferred from A5, as its README states.
Actual hardware routing must be checked against the new model logs.

The supplement contains 341 kernels: 214 II probes and 127 forwarding probes.
These cover the 152 missing forms plus 62 native-stage forms needed to separate
compound APIs. Of the 341 kernels, 61 are compound-API inspection probes; they
do not yield a single-instruction timing. Native-stage probes are separate files.
The original 140-kernel suite is unchanged and is not rerun automatically.

## Missing coverage identified

| Category | Missing probes added |
| --- | --- |
| Floating arithmetic | VLN/VSQRT/VEXPDIF FP16 |
| Reductions | Integer VCMAX/VCMIN/VCADD/VCGMAX; VCGMIN/VCGADD forms; VCPADD FP16/FP32 |
| Broadcast | Additional VDUP types; VDUP_HIGHEST, VDUP_MERGE, VDUP_HIGHEST_MERGE |
| Rearrangement | VINTLV, VDINTLV, VSELR, VSQZ, VUSQZ |
| Pack/unpack | Remaining VPACK forms; VUNPACK signed/unsigned widening forms |
| Predicates | PINTLV/PDINTLV b8/b16/b32; PPACK/PUNPACK b8 |
| Histograms | CHISTV2/DHISTV2 uint16 output, uint8 input |

Exact types and generated filenames are in `op_kernel/a6_exu2/manifest.json`,
under `coverage`. Each entry is `existing_probe`, `new_probe`, or
`native_stage_extra`. Existing forms have no duplicate supplement source.

## Run on the A6 server

Use the same two-package environment that successfully ran VADD. Compiler and
toolkit come from CANN_HOME; the full-dump model comes from FULL_SIMULATOR_HOME.
Do not source the second package's environment script.

```bash
export CANN_HOME=/absolute/path/to/compiler/toolkit
export FULL_SIMULATOR_HOME=/absolute/path/to/debug/tools/simulator
source benchmarks/a6/set_env.sh

# Inspect the supplement without compiling or simulating.
python3 run_a6_bench.py --cases op_kernel/a6_exu2 --list

# First smoke test: six additional arithmetic kernels with numerical checks.
python3 run_a6_bench.py --cases op_kernel/a6_exu2 \
  --ops VLN VSQRT VEXPDIF --forms fp16 --output results/a6_exu2_smoke

# All supplementary probes, sequential, no timeout by default.
python3 run_a6_bench.py --cases op_kernel/a6_exu2 --output results/a6_exu2

# Resume only after the previous process has exited.
python3 run_a6_bench.py --cases op_kernel/a6_exu2 \
  --output results/a6_exu2 --resume
```

To run a smaller group, add e.g. `--ops VINTLVV2 VDINTLVV2 --forms fp32`.
The default remains one run per case, device 0, block_dim 1, unlimited wall time.
`--timeout 3600` limits compilation plus simulation for each case to one hour.

Outputs follow the original batch layout:

```text
results/a6_exu2/summary.csv
results/a6_exu2/summary.json
results/a6_exu2/<OP>/<case>/r0/<case>__r0__core0.veccore0.*.dump
results/a6_exu2/_work/<case>/r0/measurement.json
results/a6_exu2/_work/<case>/r0/debug.log
results/a6_exu2/_work/<case>/r0/model.log
```

Only the six requested core0.veccore0 dump types are archived. Other dumps are
removed after this new run finishes unless `--keep-all-dumps` is specified.
`measurement.json` now includes `streams[].observed_target_exus`, extracted from
EXU events. An empty list means missing evidence, not "the operation uses EXU0".
These observations do not automatically rewrite VfSimulator's routing/timings.

## Probe semantics and limits

- II uses eight independent target calls in a loop, two iterations by default.
  Forwarding uses A0,A1,B0,B1,C0,C1 with two real RAW chains. Latency is collected
  from the same logs, not from a separate kernel or wall-clock measurement.
- Reduction compatibility APIs and two-output interleave APIs may expand into
  multiple instructions. Their `compound_api` status is intentional. Measure
  VCG*V2, VC*V2, VCPADDV2, VPACKV2 and interleave V2 probes separately. VCPADD's
  additional packing stage is not equivalent to VCPADDV2 itself.
- Interleave native probes currently select LOWER. The compound probes retain
  both LOWER and HIGHER outputs. They do not separately calibrate HIGHER timing.
  The local compiler lacks the f8e8m0 two-output wrapper, so its compound probe
  explicitly invokes both native halves, rather than casting to another dtype.
- VDUP uses a partial mask so zeroing and merging are distinguishable. Merge
  destinations are initialized; the parser includes their implicit destination
  read when checking dependence. Highest variants use the VDUPM opcode alias.
- VSELR indices are masked to the vector's valid lane range. VUSQZ destinations
  start at zero and masks depend on input data; histogram accumulators also start
  at zero. VUSQZ/histograms currently have II probes only. Predicate and
  input/output-type-changing forms also have no self-forwarding claim here.
- Most new kernels transport input/output as raw uint32 words, casting vector
  bits to the requested types without VCVT. This includes FP8/BF16 rearrangement
  tests. `semantic_input_dtype` and `semantic_output_dtype` record the actual
  instruction types; MB_TENSOR's uint32 describes transport, not computation.
- These raw-bit probes report `golden=not_implemented`. They are not numerical
  correctness tests. Reduction/pack/squeeze outputs can have undefined inactive
  lanes; do not compare their entire raw output against an ordinary elementwise
  golden. The FP16 VLN/VSQRT/VEXPDIF probes reuse the arithmetic golden framework.
- Compiler folding/reordering can still affect probes. Missing target counts,
  absent RAW edges or absent EXU issue evidence are reported as invalid/unknown,
  never converted into zero latency or assumed II. Inspect actual dumps before
  accepting a measured interval as a hardware limit.

## Validation and regeneration

All 341 current source hashes have a successful local CCEC compile/link check
for `dav-920r1-vec` (CANN 9.0.0 clang 15.0.5, build 2026-04-25).
The unit suite has 43 passing tests. There is no runnable local A6 simulator;
these new probes have NOT yet been validated end-to-end on A6, and no new A6
timing values are claimed. Recompile with the remote compiler when running.

```bash
# Recreate from the checked-in inventory, no VfSimulator checkout required.
python3 -m benchmarks.a6.exu2
# Refresh the inventory from a new DV121 configuration explicitly.
python3 -m benchmarks.a6.exu2 --isa /path/to/configs/socs/DV121/isa.json
python3 -m unittest benchmarks.a6.test_suite benchmarks.a6.test_exu2 -v
# Compile-only preflight; this does not launch camodel.
python3 -m benchmarks.a6.check_compile --cases op_kernel/a6_exu2 \
  --cann-home "$CANN_HOME" --output results/exu2_compile_check --jobs 4
```

Regeneration refuses to overwrite previously generated files that were manually
edited. Use a different `--output` directory for exploratory source changes.
