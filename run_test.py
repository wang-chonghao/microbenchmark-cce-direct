#!/usr/bin/env python3
import argparse
import ast
import datetime as dt
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

SUPPORTED_TENSOR_DTYPES = {
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "fp16", "bf16", "fp32", "float32", "fp64", "float64",
}


@dataclass(frozen=True)
class TensorSpec:
    direction: str
    name: str
    dtype: str
    elements: int


@dataclass(frozen=True)
class KernelManifest:
    kernel_name: str
    tensors: tuple[TensorSpec, ...]
    goldens: tuple[tuple[str, str], ...]


def parse_kernel_manifest(kernel_path: Path, total_count: int) -> KernelManifest | None:
    """Parse MB_KERNEL/MB_TENSOR declarations embedded in a CCE source file."""
    text = kernel_path.read_text(encoding="utf-8")
    kernel_matches = re.findall(r"^\s*//\s*MB_KERNEL\s+([A-Za-z_]\w*)\s*$", text, re.MULTILINE)
    tensor_matches = re.findall(
        r"^\s*//\s*MB_TENSOR\s+(input|output|inout)\s+([A-Za-z_]\w*)\s+"
        r"([A-Za-z0-9_]+)\s+([A-Za-z0-9_xX]+)\s*$",
        text,
        re.MULTILINE,
    )
    golden_matches = re.findall(
        r"^\s*//\s*MB_GOLDEN\s+([A-Za-z_]\w*)\s+(.+?)\s*$",
        text,
        re.MULTILINE,
    )
    if not kernel_matches and not tensor_matches:
        return None
    if len(kernel_matches) != 1:
        raise RuntimeError("generic mode requires exactly one '// MB_KERNEL <symbol>' declaration")
    if not tensor_matches:
        raise RuntimeError("generic mode requires at least one '// MB_TENSOR ...' declaration")

    tensors = []
    names = set()
    for direction, name, dtype, shape in tensor_matches:
        dtype = {"float32": "fp32", "float64": "fp64"}.get(dtype, dtype)
        if dtype not in SUPPORTED_TENSOR_DTYPES:
            supported = ", ".join(sorted(SUPPORTED_TENSOR_DTYPES))
            raise RuntimeError(f"unsupported dtype '{dtype}' in MB_TENSOR; supported: {supported}")
        if name in names:
            raise RuntimeError(f"duplicate MB_TENSOR name: {name}")
        names.add(name)

        if shape == "TOTAL_COUNT":
            elements = total_count
        else:
            dims = re.split(r"[xX]", shape)
            if not dims or any(not dim.isdigit() or int(dim) <= 0 for dim in dims):
                raise RuntimeError(
                    f"invalid shape '{shape}' for tensor {name}; use a positive count, e.g. 256 or 8x1024"
                )
            elements = 1
            for dim in dims:
                elements *= int(dim)
        tensors.append(TensorSpec(direction, name, dtype, elements))

    tensor_names = {item.name for item in tensors}
    for output_name, _ in golden_matches:
        if output_name not in tensor_names:
            raise RuntimeError(f"MB_GOLDEN references unknown tensor: {output_name}")
    return KernelManifest(kernel_matches[0], tuple(tensors), tuple(golden_matches))


def write_tensor_spec(manifest: KernelManifest, path: Path) -> None:
    lines = [f"{item.direction}\t{item.name}\t{item.dtype}\t{item.elements}\n" for item in manifest.tensors]
    path.write_text("".join(lines), encoding="utf-8")


def dtype_format(dtype: str) -> str:
    mapping = {
        "int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
        "int32": "i", "uint32": "I", "int64": "q", "uint64": "Q",
        "fp32": "f", "float32": "f", "fp64": "d", "float64": "d",
    }
    if dtype not in mapping:
        raise RuntimeError(f"golden check does not support dtype: {dtype}")
    return mapping[dtype]


def read_tensor_values(path: Path, dtype: str, elements: int) -> list[float]:
    data = path.read_bytes()
    fmt = dtype_format(dtype)
    size = struct_size(fmt)
    expected = elements * size
    if len(data) != expected:
        raise RuntimeError(f"unexpected file size for {path}: got {len(data)}, expected {expected}")
    import struct
    return list(struct.unpack("<" + fmt * elements, data))


def struct_size(fmt: str) -> int:
    import struct
    return struct.calcsize("<" + fmt)


class SafeExpr:
    def __init__(self, expr: str):
        self.expr = expr
        self.tree = ast.parse(expr, mode="eval")
        self._validate(self.tree)

    def _validate(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
            ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
            ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or,
        )
        if not isinstance(node, allowed):
            raise RuntimeError(f"unsupported MB_GOLDEN syntax in '{self.expr}': {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            self._validate(child)

    def eval(self, env: dict[str, float], arrays: dict[str, list[float]], index: int) -> float:
        def at(name, item):
            pos = int(item)
            values = arrays[name]
            if pos < 0 or pos >= len(values):
                raise RuntimeError(f"MB_GOLDEN at({name}, {pos}) out of range")
            return values[pos]

        def where(cond, true_value, false_value):
            return true_value if cond else false_value

        safe_globals = {
            "__builtins__": {},
            "exp": math.exp,
            "sqrt": math.sqrt,
            "abs": abs,
            "min": min,
            "max": max,
            "where": where,
            "at": at,
        }
        safe_locals = dict(env)
        safe_locals["idx"] = index
        return eval(compile(self.tree, "<MB_GOLDEN>", "eval"), safe_globals, safe_locals)


def check_generic_goldens(manifest: KernelManifest, run_dir: Path, atol: float, rtol: float) -> None:
    check_log = run_dir / "golden_check.log"

    def log_check(message: str) -> None:
        print(message)
        with check_log.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    check_log.write_text("", encoding="utf-8")
    if not manifest.goldens:
        log_check("[CHECK] skipped (no MB_GOLDEN declarations)")
        return

    specs = {item.name: item for item in manifest.tensors}
    arrays: dict[str, list[float]] = {}
    for spec in manifest.tensors:
        if spec.dtype in ("fp16", "bf16"):
            raise RuntimeError("MB_GOLDEN currently supports fp32/fp64/integer tensors, not fp16/bf16")
        if spec.direction == "inout":
            input_path = run_dir / f"input_{spec.name}.bin"
            output_path = run_dir / f"output_{spec.name}.bin"
            arrays[f"{spec.name}_in"] = read_tensor_values(input_path, spec.dtype, spec.elements)
            arrays[spec.name] = read_tensor_values(output_path, spec.dtype, spec.elements)
            continue
        if spec.direction == "input":
            path = run_dir / f"input_{spec.name}.bin"
        else:
            path = run_dir / f"output_{spec.name}.bin"
        arrays[spec.name] = read_tensor_values(path, spec.dtype, spec.elements)

    for output_name, expr in manifest.goldens:
        spec = specs[output_name]
        if spec.direction not in ("output", "inout"):
            raise RuntimeError(f"MB_GOLDEN target is not an output/inout tensor: {output_name}")
        output = arrays[output_name]
        compiled = SafeExpr(expr)
        max_abs = 0.0
        max_rel = 0.0
        bad_index = None
        for i, actual in enumerate(output):
            env = {}
            for name, values in arrays.items():
                if i < len(values):
                    env[name] = values[i]
            expected = compiled.eval(env, arrays, i)
            abs_err = abs(float(actual) - float(expected))
            rel_err = abs_err / max(abs(float(expected)), 1.0)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if abs_err > atol + rtol * abs(float(expected)):
                bad_index = i
                log_check(
                    f"[CHECK] FAIL {output_name}[{i}] actual={actual:.10g} "
                    f"expected={expected:.10g} abs={abs_err:.3g} rel={rel_err:.3g}"
                )
                break
        if bad_index is not None:
            raise RuntimeError(f"golden check failed for {output_name}: {expr}")
        log_check(f"[CHECK] PASS {output_name}: {expr} max_abs={max_abs:.3g} max_rel={max_rel:.3g}")


def default_cann_home() -> str:
    env = os.environ.get("CANN_HOME")
    if env:
        return env
    local = Path("/home/lenovo/.codex/memories/cann-9.0.0/cann-9.0.0")
    if local.exists():
        return str(local)
    return ""


def default_arch() -> str:
    env = os.environ.get("ARCH")
    if env:
        return env
    machine = platform.machine()
    if machine == "x86_64":
        return "x86_64-linux"
    if machine in ("aarch64", "arm64"):
        return "aarch64-linux"
    return f"{machine}-linux"


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def quote(value) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def run_bash(script: str, cwd: Path, log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(proc.stdout)
    if proc.stdout:
        print(proc.stdout, end="")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}")
    return proc


def require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{what} not found: {path}")
    return path


def cce_include_flags(cann_home: Path, arch: str) -> list[str]:
    candidates = [
        Path("/usr/include/c++/11"),
        Path("/usr/include/c++/12"),
        Path("/usr/include/c++/13"),
        Path("/usr/include/x86_64-linux-gnu/c++/11"),
        Path("/usr/include/x86_64-linux-gnu/c++/12"),
        Path("/usr/include/x86_64-linux-gnu/c++/13"),
        Path("/usr/include/aarch64-linux-gnu/c++/11"),
        Path("/usr/include/aarch64-linux-gnu/c++/12"),
        Path("/usr/include/aarch64-linux-gnu/c++/13"),
        cann_home / "include",
        cann_home / arch / "include",
        cann_home / arch / "asc",
        cann_home / arch / "asc" / "include",
        cann_home / arch / "asc" / "include" / "basic_api",
        cann_home / arch / "asc" / "include" / "interface",
        cann_home / arch / "asc" / "impl",
        cann_home / arch / "asc" / "impl" / "basic_api",
        cann_home / arch / "pkg_inc",
    ]
    flags = []
    for item in candidates:
        if item.is_dir():
            flags.append(f"-I{item}")
    return flags


def find_tool(cann_home: Path, arch: str, name: str) -> str:
    candidates = [
        cann_home / arch / "bin" / name,
        cann_home / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} not found. Source set_env.sh or check CANN_HOME/ARCH.")


def compile_kernel(args, cann_home: Path, arch: str, build_dir: Path, log_path: Path) -> Path:
    kernel_src = args.kernel.resolve()
    require_file(kernel_src, "CCE kernel")
    ccec = find_tool(cann_home, arch, "ccec")
    ld_lld = find_tool(cann_home, arch, "ld.lld")
    aiv_obj = build_dir / f"{args.kernel_name}_mix_aiv.o"
    kernel_bin = build_dir / f"{args.kernel_name}_mix.o"
    includes = " ".join(quote(flag) for flag in cce_include_flags(cann_home, arch))
    extra_flags = os.environ.get("CCEC_EXTRA_FLAGS", "")
    script = f"""
set -e
source {quote(cann_home / 'set_env.sh')}
export ASCEND_HOME_PATH={quote(cann_home)}
export ASCEND_CANN_PACKAGE_PATH={quote(cann_home)}
{quote(ccec)} -g -std=c++17 -c -O2 {quote(kernel_src)} -o {quote(aiv_obj)} \
  {includes} \
  --cce-aicore-arch={quote(args.core_arch)} \
  --cce-aicore-only \
  -mllvm -cce-aicore-function-stack-size=16000 \
  -mllvm -cce-aicore-record-overflow=false \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-jump-expand=true \
  --cce-simd-vf-fusion=false \
  -DTOTAL_COUNT={args.total_count} \
  {extra_flags}
{quote(ld_lld)} -Ttext=0 {quote(aiv_obj)} -static -o {quote(kernel_bin)}
"""
    run_bash(script, cwd=REPO_ROOT, log_path=log_path)
    return kernel_bin


def compile_runner(args, cann_home: Path, arch: str, build_dir: Path, log_path: Path,
                   generic_mode: bool) -> Path:
    runner_name = "generic_runner.cpp" if generic_mode else "native_runner.cpp"
    runner_src = REPO_ROOT / "runner" / runner_name
    runner_bin = build_dir / "native_cce_runner"
    host_inc = cann_home / arch / "include"
    pkg_inc = cann_home / arch / "pkg_inc"
    cann_lib = cann_home / arch / "lib64"
    sim_lib = cann_home / arch / "simulator" / args.soc_version / "lib"
    dav_lib = cann_home / arch / "simulator" / "dav_3510" / "lib"
    dav_camodel = cann_home / arch / "simulator" / "dav_3510" / "camodel"
    devlib = cann_home / arch / "devlib"
    devlib_device = cann_home / arch / "devlib" / "device"
    device_lib = cann_home / arch / "lib64" / "device" / "lib64"
    script = f"""
set -e
source {quote(cann_home / 'set_env.sh')}
g++ -std=c++17 -O2 -Wl,-z,relro -Wl,-z,now -Wl,--allow-shlib-undefined \
  -o {quote(runner_bin)} {quote(runner_src)} \
  -I{quote(host_inc)} \
  -I{quote(host_inc / 'experiment' / 'msprof')} \
  -I{quote(host_inc / 'experiment' / 'msprof' / 'toolchain')} \
  -I{quote(cann_home / 'include')} \
  -I{quote(pkg_inc)} \
  -I{quote(pkg_inc / 'runtime')} \
  -I{quote(pkg_inc / 'profiling')} \
  -I{quote(cann_home / arch / 'pkg_inc' / 'toolchain')} \
  -L{quote(cann_lib)} \
  -L{quote(sim_lib)} \
  -L{quote(dav_lib)} \
  -L{quote(dav_camodel)} \
  -L{quote(devlib)} \
  -Wl,-rpath,{quote(str(cann_lib) + ':' + str(sim_lib) + ':' + str(dav_lib) + ':' + str(dav_camodel) + ':' + str(devlib) + ':' + str(devlib_device) + ':' + str(device_lib) + ':' + str(build_dir))} \
  -lruntime_camodel -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
"""
    run_bash(script, cwd=REPO_ROOT, log_path=log_path)
    return runner_bin


def runtime_env_script(cann_home: Path, arch: str, args, run_dir: Path) -> str:
    host_machine = platform.machine()
    host_devlib = "aarch64" if host_machine in ("aarch64", "arm64") else "x86_64"
    other_devlib = "x86_64" if host_devlib == "aarch64" else "aarch64"
    full_simulator_home = os.environ.get("FULL_SIMULATOR_HOME", "").strip()
    paths = [
        cann_home / "lib64",
        cann_home / "fwkacllib" / "lib64",
        cann_home / "runtime" / "lib64",
        cann_home / arch / "lib64",
        cann_home / arch / "lib64" / "device" / "lib64",
        cann_home / arch / "devlib",
        cann_home / arch / "devlib" / "device",
        cann_home / arch / "devlib" / "linux" / host_devlib,
        cann_home / arch / "devlib" / "linux" / other_devlib,
        cann_home / arch / "simulator" / "dav_3510" / "camodel",
        cann_home / arch / "simulator" / "dav_3510" / "lib",
        cann_home / "tools" / "simulator" / args.soc_version / "lib",
        cann_home / arch / "simulator" / args.soc_version / "lib",
    ]
    if full_simulator_home:
        full_simulator = Path(full_simulator_home).expanduser().resolve()
        paths = [
            full_simulator / args.soc_version / "camodel",
            full_simulator / args.soc_version / "lib",
            full_simulator / "dav_3510" / "camodel",
            full_simulator / "dav_3510" / "lib",
        ] + paths
    ld = ":".join(str(p) for p in paths if p.exists())
    script = f"""
source {quote(cann_home / 'set_env.sh')}
export ASCEND_HOME_PATH={quote(cann_home)}
export ASCEND_CANN_PACKAGE_PATH={quote(cann_home)}
export ASCEND_TOOLKIT_HOME={quote(cann_home)}
export ASCEND_OPP_PATH={quote(cann_home / 'opp')}
export ASCEND_DEVICE_ID={args.device_id}
export ACL_DEVICE_ID={args.device_id}
export ASCEND_PROCESS_LOG_PATH={quote(run_dir / 'ascend_process_log')}
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
export LD_LIBRARY_PATH={quote(ld)}:${{LD_LIBRARY_PATH:-}}
unset LD_PRELOAD
"""
    if full_simulator_home:
        script += f"export FULL_SIMULATOR_HOME={quote(Path(full_simulator_home).expanduser().resolve())}\n"
    return script


def copy_camodel_config(cann_home: Path, arch: str, args, run_dir: Path) -> None:
    etc = run_dir / "etc"
    etc.mkdir(parents=True, exist_ok=True)
    candidates = []
    env_cfg = os.environ.get("CAMODEL_CONFIG_TOML")
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates += [
        cann_home / arch / "simulator" / "dav_3510" / "lib" / "1982_cloud_config.toml",
        cann_home / arch / "simulator" / args.soc_version / "lib" / "1982_cloud_config.toml",
        cann_home / "tools" / "simulator" / args.soc_version / "lib" / "1982_cloud_config.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, etc / "1982_cloud_config.toml")
            print(f"[INFO] copied camodel config: {candidate}")
            return
    print("[WARN] 1982_cloud_config.toml not found; set CAMODEL_CONFIG_TOML if core_wrapper needs it")


def run_kernel(args, cann_home: Path, arch: str, out_dir: Path, kernel_bin: Path, runner_bin: Path,
               log_path: Path, tensor_spec: Path | None, manifest: KernelManifest | None) -> None:
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "log_ca").mkdir(parents=True, exist_ok=True)
    (run_dir / "log" / "ub_log").mkdir(parents=True, exist_ok=True)
    (run_dir / "log" / "gm_log").mkdir(parents=True, exist_ok=True)
    for item in (out_dir, run_dir):
        try:
            item.chmod(0o700)
        except PermissionError:
            pass
    for item in (run_dir / "log", run_dir / "log" / "ub_log", run_dir / "log" / "gm_log"):
        try:
            item.chmod(0o700)
        except PermissionError:
            pass
    shutil.copy2(kernel_bin, run_dir / kernel_bin.name)
    shutil.copy2(runner_bin, run_dir / runner_bin.name)
    if tensor_spec is not None:
        shutil.copy2(tensor_spec, run_dir / tensor_spec.name)
    os.chmod(run_dir / runner_bin.name, 0o755)
    copy_camodel_config(cann_home, arch, args, run_dir)

    env_script = runtime_env_script(cann_home, arch, args, run_dir)
    if tensor_spec is not None:
        app = (
            f"./{runner_bin.name} ./{kernel_bin.name} {args.kernel_name} ./{tensor_spec.name} "
            f"{args.block_dim} {args.local_memory_size}"
        )
    else:
        app = (
            f"./{runner_bin.name} ./{kernel_bin.name} {args.kernel_name} {args.dtype} "
            f"{args.total_count} {args.block_dim} {args.local_memory_size} {args.golden} {args.atol} {args.rtol}"
        )
    command = f"""
set -e
{env_script}
cd {quote(run_dir)}
{app}
"""
    run_bash(command, cwd=REPO_ROOT, log_path=log_path)
    if manifest is not None and args.golden != "none":
        check_generic_goldens(manifest, run_dir, args.atol, args.rtol)


def summarize(out_dir: Path) -> None:
    print(f"[INFO] OUTPUT={out_dir}")
    for path in [
        out_dir / "build",
        out_dir / "run",
    ]:
        if path.exists():
            print(f"[INFO] {path.name}={path}")
    dump_files = sorted((out_dir / "run" / "log_ca").glob("*.dump")) if (out_dir / "run" / "log_ca").exists() else []
    if dump_files:
        print("[INFO] log_ca dump samples:")
        for item in dump_files[:20]:
            print(f"  log_ca/{item.name} {item.stat().st_size} bytes")
    for item in sorted((out_dir / "run").glob("output_*.bin"))[:20]:
        print(f"[INFO] output={item} {item.stat().st_size} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and directly run a CCE microbenchmark on CANN camodel runtime.")
    parser.add_argument("--cann-home", default=default_cann_home())
    parser.add_argument("--arch", default=default_arch())
    parser.add_argument("--soc-version", default=os.environ.get("SOC_VERSION", "Ascend950PR_9599"))
    parser.add_argument("--core-arch", default=os.environ.get("CORE_ARCH", "dav-c310-vec"))
    parser.add_argument("--kernel", type=Path, default=REPO_ROOT / "op_kernel" / "kernel.cce")
    parser.add_argument("--kernel-name", default="foo_add")
    parser.add_argument("--dtype", choices=("int32", "fp32", "float32"), default="int32")
    parser.add_argument("--total-count", type=int, default=256)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--local-memory-size", type=int, default=int(os.environ.get("LOCAL_MEMORY_SIZE", "0")),
                        help="Dynamic UB/local memory size in bytes. 0 keeps the runtime default.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--golden", choices=("none", "add", "exp_mul"), default="add")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--compile-only", action="store_true", help="Only compile the kernel and native runner.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cann_home = Path(args.cann_home).expanduser().resolve()
    if not cann_home.exists():
        raise RuntimeError(f"CANN_HOME/--cann-home does not exist: {cann_home}")
    if args.total_count <= 0:
        raise RuntimeError("--total-count must be positive")
    if args.block_dim <= 0:
        raise RuntimeError("--block-dim must be positive")
    if args.local_memory_size < 0:
        raise RuntimeError("--local-memory-size must be non-negative")

    out_dir = (args.output or (REPO_ROOT / "result" / timestamp())).resolve()
    build_dir = out_dir / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    print(f"CANN_HOME={cann_home}")
    print(f"ARCH={args.arch}")
    print(f"OUTPUT={out_dir}")

    kernel_path = args.kernel.expanduser().resolve()
    require_file(kernel_path, "CCE kernel")
    manifest = parse_kernel_manifest(kernel_path, args.total_count)
    tensor_spec = None
    if manifest is not None:
        args.kernel_name = manifest.kernel_name
        tensor_spec = build_dir / "kernel_tensors.tsv"
        write_tensor_spec(manifest, tensor_spec)
        print(f"[INFO] generic tensor mode: kernel={manifest.kernel_name}, args={len(manifest.tensors)}")
        for index, item in enumerate(manifest.tensors):
            print(f"  arg[{index}] {item.direction} {item.name}: {item.dtype}[{item.elements}]")
        if args.golden != "none" and not manifest.goldens:
            print("[INFO] generic tensor mode has no MB_GOLDEN declarations")
    else:
        print("[INFO] legacy three-tensor mode (no MB_KERNEL/MB_TENSOR declarations found)")

    kernel_bin = compile_kernel(args, cann_home, args.arch, build_dir, log_path)
    runner_bin = compile_runner(args, cann_home, args.arch, build_dir, log_path, manifest is not None)
    if args.compile_only:
        print(f"[INFO] kernel_bin={kernel_bin}")
        print(f"[INFO] runner_bin={runner_bin}")
        summarize(out_dir)
        return 0
    run_kernel(args, cann_home, args.arch, out_dir, kernel_bin, runner_bin, log_path, tensor_spec, manifest)
    summarize(out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
