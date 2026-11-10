"""Compile unchanged A6 probe kernels; optionally run with a user-supplied A6 CAModel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(argv, folder, name, env=None):
    with (folder/(name+'.log')).open('w') as stream:
        result = subprocess.run(list(map(str, argv)), cwd=folder, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    (folder/(name+'.command.json')).write_text(json.dumps(dict(argv=list(map(str, argv)),
        returncode=result.returncode), indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {folder/(name+".log")}')


def compiler_command(compiler, source, output):
    return [str(compiler), '-g', '-std=c++17', '-c', '-O2', str(source), '-o', str(output),
            '--cce-aicore-arch=dav-920r1-vec', '--cce-aicore-only', '--cce-simd-vf-fusion=false',
            '-mllvm', '-cce-aicore-vec-misched=0', '-mllvm', '-cce-vf-stack-size=0']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolkit', type=Path, default=os.environ.get('MEMBAR_CANN_ROOT') or os.environ.get('ASCEND_TOOLKIT_HOME'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=json.loads((HERE/'experiments.json').read_text())['cases'])
    parser.add_argument('--run', action='store_true', help='Run only if this SDK has the matching A6 simulator')
    parser.add_argument('--npu-type', help='Exact installed A6 simulator name; no A5 fallback')
    parser.add_argument('--host-arch', default=os.environ.get('MEMBAR_HOST_ARCH', platform.machine()+'-linux'))
    args = parser.parse_args()
    if args.toolkit is None:
        parser.error('Specify --toolkit or MEMBAR_CANN_ROOT')
    toolkit, output = args.toolkit.absolute(), args.out.absolute()
    if output.exists() and any(output.iterdir()):
        parser.error('--out must be a new or empty directory')
    if args.run and not args.npu_type:
        parser.error('--run requires the actual A6 --npu-type')

    def tool(name, override):
        if os.environ.get(override):
            return Path(os.environ[override])
        return next((p for p in (toolkit/'tools/bisheng_compiler/bin'/name,
                                toolkit/'bin'/name, toolkit/'x86_64-linux/bin'/name) if p.is_file()),
                    toolkit/'bin'/name)

    ccec, linker = tool('ccec', 'MEMBAR_CCEC'), tool('ld.lld', 'MEMBAR_LINKER')
    simulator = next((p for p in (toolkit/'tools/simulator'/str(args.npu_type)/'lib',
                                  toolkit/args.host_arch/'simulator'/str(args.npu_type)/'lib')
                      if (p/'libruntime_camodel.so').is_file()), toolkit/'tools/simulator'/str(args.npu_type)/'lib')
    if args.run and not (simulator/'libruntime_camodel.so').is_file():
        parser.error(f'Matching simulator library not found: {simulator}; compile-only is still available')
    manifest = json.loads((HERE/'experiments.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name in args.cases or manifest['cases']:
        folder = output/name
        source = folder/'source'
        source.mkdir(parents=True)
        for file in (name+'.cce', 'probe_common.h', 'host.cpp'):
            shutil.copy2(HERE/file, source/file)
        execute(compiler_command(ccec, source/(name+'.cce'), folder/'kernel_aiv.o'), folder, 'compile')
        execute([linker, '-Ttext=0', 'kernel_aiv.o', '-static', '-o', 'kernel.o'], folder, 'link')
        receipt = dict(case=name, target=manifest['target'], source_sha256=sha(source/(name+'.cce')),
                       header_sha256=sha(source/'probe_common.h'), binary_sha256=sha(folder/'kernel.o'),
                       vf_stack_budget=0, compiled=True, runtime_verified=False)
        (folder/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if args.run:
            includes = [toolkit/'include'] + [toolkit/args.host_arch/p for p in
                        ('include', 'pkg_inc', 'pkg_inc/runtime', 'pkg_inc/profiling',
                         'include/experiment/msprof', 'include/experiment/msprof/toolchain')]
            includes += [Path(p) for p in os.environ.get('MEMBAR_CXX_INCLUDE_DIRS', '').split(':') if p]
            libs = [simulator, toolkit/'lib64', toolkit/args.host_arch/'lib64',
                    toolkit/args.host_arch/'devlib', toolkit/args.host_arch/'devlib/device']
            libs += [Path(p) for p in os.environ.get('MEMBAR_EXTRA_LIBRARY_DIRS', '').split(':') if p]
            execute([tool('bisheng', 'MEMBAR_HOST_CXX'), '-std=c++17', '-O2', source/'host.cpp', '-o', 'host',
                     *[f'-I{p}' for p in includes], *[f'-L{p}' for p in libs], '-Wl,--allow-shlib-undefined',
                     '-lruntime_camodel', '-lascendcl', '-lstdc++', '-ldl', '-lpthread'], folder, 'host_build')
            env = dict(os.environ, ASCEND_TOOLKIT_HOME=str(toolkit), ACL_PATH=str(toolkit), NPU_TYPE=args.npu_type)
            env['LD_LIBRARY_PATH'] = ':'.join(map(str, libs))+':'+os.environ.get('LD_LIBRARY_PATH', '')
            execute([folder/'host', folder/'kernel.o', manifest['cases'][name]['kernel']], folder, 'camodel', env)
            if (folder/'output.bin').read_bytes() != (folder/'golden.bin').read_bytes():
                raise ValueError('Host golden output mismatch')
            receipt.update(runtime_verified=True, numerical_verified=True, npu_type=args.npu_type)
            (folder/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(name, 'compiled', 'golden PASS' if args.run else 'runtime not tested', flush=True)


if __name__ == '__main__':
    main()
