"""Import reviewed byte-buffer operator wrappers; reject unknown layouts."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_index(path):
    class IndexParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.capture, self.parts = False, []

        def handle_starttag(self, tag, attrs):
            if tag == 'script':
                self.capture = dict(attrs).get('id') == 'embedded-index'

        def handle_data(self, data):
            if self.capture:
                self.parts.append(data)

        def handle_endtag(self, tag):
            if tag == 'script':
                self.capture = False
    parser = IndexParser()
    with path.open() as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), ''):
            parser.feed(chunk)
    parser.close()
    return json.loads(''.join(parser.parts))['files']


def calls(text, name):
    """Balanced argument reader for the wrapper's intrinsic calls only."""
    result = []
    for match in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        start = match.end()
        depth, args, last = 1, [], start
        for pos in range(start, len(text)):
            ch = text[pos]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    args.append(text[last:pos].strip())
                    result.append((match.start(), pos + 1, args))
                    break
            elif ch == ',' and depth == 1:
                args.append(text[last:pos].strip())
                last = pos + 1
        else:
            raise ValueError('Unbalanced wrapper call')
    return result


def adapt(text):
    match = re.search(r'extern "C" __global__ __aicore__ void candidate_kernel\s*\('
                      r'__gm__ uint8_t\* input,\s*__gm__ uint8_t\* output\)\s*\{', text)
    if not match:
        raise ValueError('Nonstandard entry; needs an individually reviewed wrapper/input layout')
    vf, wrapper = text[:match.start()], text[match.start():]
    original_vf = vf
    if 'AscendC::' in vf or 'using namespace AscendC' in vf:
        raise ValueError('AscendC API use requires separate header adaptation')
    vf = vf.replace('#include "kernel_operator.h"',
                    '#include <stdint.h>\n#ifndef __aicore__\n#define __aicore__ [aicore]\n#endif')
    if '__local_mem__' in vf:
        vf = '#ifndef __local_mem__\n#define __local_mem__ __ubuf__\n#endif\n' + vf
    loads = calls(wrapper, 'copy_gm_to_ubuf_align_v2')
    stores = calls(wrapper, 'copy_ubuf_to_gm_align_v2')
    if len(loads) != 1 or not stores:
        raise ValueError('Expected one packed input copy and nonempty outputs')
    a = loads[0][2]
    if len(a) != 11 or a[:4] != ['(__ubuf__ uint8_t*)get_imm(0)', 'input', '0', '1'] or any(x != '0' for x in a[5:]):
        raise ValueError('Unsupported input copy layout')
    input_bytes = int(a[4])
    if input_bytes <= 0 or input_bytes % 4:
        raise ValueError('Input must be whole 32-bit words')
    addresses = dict(re.findall(r'__ubuf__\s+\w+\s*\*\s*(\w+)\s*=\s*\(__ubuf__\s+\w+\s*\*\)get_imm\((\d+|0x[0-9a-fA-F]+)\)', wrapper))
    types = dict((name, dtype) for dtype, name in re.findall(r'__ubuf__\s+(\w+)\s*\*\s*(\w+)\s*=', wrapper))
    replacements = [(loads[0][0], loads[0][1],
        f'nddma_desc a6_desc(nddma_desc::loop_desc({input_bytes // 4}, 1, 1));\n'
        '    nddma_out_to_ub((__ubuf__ uint32_t*)get_imm(0), '
        '(__gm__ uint32_t*)input, 0, a6_desc, 0, NEAREST_PADDING, 0)')]
    outputs, output_bytes = [], 0
    for start, end, args in stores:
        if len(args) != 8 or args[2:4] != ['0', '1'] or args[5:] != ['0', '0', '0']:
            raise ValueError('Unsupported output copy layout')
        offset = re.fullmatch(r'output\s*\+\s*(\d+)', args[0])
        source = re.fullmatch(r'\(__ubuf__ uint8_t\*\)(\w+)', args[1])
        if not offset or not source or source[1] not in addresses:
            raise ValueError('Unknown output address')
        offset, size = int(offset[1]), int(args[4])
        ub = int(addresses[source[1]], 0)
        if offset != output_bytes or size <= 0 or size % 4 or ub % 32:
            raise ValueError('Noncontiguous output or unsafe scatter-offset alias/alignment')
        output_bytes += size
        outputs.append(dict(name=source[1], gm_offset=offset, bytes=size, ub_start=ub, dtype=types[source[1]]))
        replacements.append((start, end,
            f'scatter_ubuf_to_gm((__gm__ uint32_t*)(output + {offset}), '
            f'(__ubuf__ uint32_t*){source[1]}, 0, 0, a6_offsets, 0, {size}, 0, 1)'))
    # After VF completes its temporaries are dead. Choose a 32-byte range not
    # overlapping any live copy-out buffer, including in-place outputs at zero.
    offsets_address = 0
    for output in sorted(outputs, key=lambda o: o['ub_start']):
        if offsets_address + 32 > output['ub_start']:
            offsets_address = max(offsets_address, (output['ub_start'] + output['bytes'] + 31) // 32 * 32)
    if offsets_address + 32 > 248 * 1024:
        raise ValueError('Scatter offsets exceed conservative UB budget')
    first = stores[0][0]
    prelude = ('pipe_barrier(PIPE_ALL);\n'
        '    set_flag(PIPE_V, PIPE_S, EVENT_ID1);\n    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);\n'
        f'    __ubuf__ uint32_t* a6_offsets = (__ubuf__ uint32_t*)get_imm({offsets_address});\n'
        '    a6_offsets[0] = 0;\n'
        '    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);\n    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);\n    ')
    for start, end, value in sorted(replacements, reverse=True):
        wrapper = wrapper[:start] + (prelude if start == first else '') + value + wrapper[end:]
    close = wrapper.rfind('}')
    wrapper = wrapper[:close] + ('    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);\n'
                               '    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);\n') + wrapper[close:]
    header = (f'// MB_KERNEL candidate_kernel\n// MB_TENSOR input input uint8 {input_bytes}\n'
              f'// MB_TENSOR output output uint8 {output_bytes}\n'
              '// A6 wrapper migration only; original VF below is unchanged.\n')
    return header + vf + wrapper, dict(input_bytes=input_bytes, output_bytes=output_bytes,
        outputs=outputs, vf_sha256=digest(vf.encode()), original_vf_sha256=digest(original_vf.encode()),
        header_adaptation='Replace unused kernel_operator.h with stdint and aicore macro; VF bodies unchanged',
        offsets_ub=[offsets_address, offsets_address + 32])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.report_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    index = archive_index(root/'spill_candidate_alignment_standalone.html')
    for name in ('spill_candidate_alignment.json', 'software_cce_links.json'):
        rel = 'metadata/' + name
        if digest((root/rel).read_bytes()) != index[rel]['sha256']:
            raise ValueError('External metadata differs from standalone report: ' + rel)
    rows = json.loads((root/'metadata/spill_candidate_alignment.json').read_text())['rows']
    links = json.loads((root/'metadata/software_cce_links.json').read_text())['rows']
    cases, excluded = [], []
    for row in rows:
        paths = [('original', row['source_cce_path'])]
        paths += [('software', item['path']) for item in links.get(row['id'], {}).get('A6', [])]
        for role, path in paths:
            name = row['id'] + '_' + role
            try:
                original = (root / path).read_bytes()
                if path not in index or digest(original) != index[path]['sha256']:
                    raise ValueError('Source missing/different in standalone report snapshot')
                generated, layout = adapt(original.decode())
                input_path = (root/row['source_cce_path']).parent/'input.bin'
                data = input_path.read_bytes()
                if len(data) != layout['input_bytes']:
                    raise ValueError('Archived input size does not match wrapper')
                directory = out/name
                directory.mkdir()
                (directory/'kernel.cce').write_text(generated)
                (directory/'original.cce').write_bytes(original)
                (directory/'input_input.bin').write_bytes(data)
                cases.append(dict(name=name, family=row['case'], row_id=row['id'], role=role,
                    source=f'{name}/kernel.cce', source_sha256=digest(generated.encode()),
                    origin=path, origin_sha256=digest(original), input=f'{name}/input_input.bin',
                    input_sha256=digest(data), golden='not_implemented',
                    compiler_flags=['-mllvm', '-cce-aicore-vec-misched=0'] +
                        (['-mllvm', '-cce-vf-stack-size=8192'] if name == 'lstm-u7_original' else []),
                    **layout))
            except (ValueError, OSError) as exc:
                excluded.append(dict(name=name, origin=path, reason=str(exc)))
    manifest = dict(cases=cases, excluded=excluded, report_rows=len(rows),
        report='spill_candidate_alignment_standalone.html',
        metadata_sha256={name:digest((root/'metadata'/name).read_bytes()) for name in
                         ('spill_candidate_alignment.json', 'software_cce_links.json')},
        note='A6 compiler validation required. Preserved VF; no numerical golden claim.')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Imported {len(cases)}, deferred {len(excluded)}')


if __name__ == '__main__':
    main()
