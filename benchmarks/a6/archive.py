"""Archive only the requested veccore0 dumps from a newly created experiment."""

import os
from pathlib import Path
import re

KINDS = ("instr_popped_log", "instr_log", "EXU", "IDU", "ISU", "OOO")
DUMP = re.compile(r"^core\d+\.veccore0\.(?:(instr_popped_log|instr_log)|rvec\.(EXU|IDU|ISU|OOO))"
                  r"(?:\.\d+)?\.dump(?:\.\d+)?$")
ANY_DUMP = re.compile(r"\.dump(?:\.\d+)?$")


def archive_dumps(run: Path, out: Path, case: dict, keep_all=False):
    """Move requested files; prune other dumps only inside this run, after model exit.

    No symlink is followed, and destination collisions fail before any deletion.
    Paths in the record are relative to the experiment directory so copying the
    complete results directory to another machine preserves analysis.
    """
    destination = out / case["op"] / case["name"] / run.name
    candidates, discarded = [], []
    for base, dirs, files in os.walk(run / "run", followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
        for name in sorted(files):
            source = Path(base) / name
            if source.is_symlink() or not ANY_DUMP.search(name):
                continue
            match = DUMP.fullmatch(name)
            if match:
                target = destination / f"{case['name']}__{run.name}__{name}"
                candidates.append((source, target, match[1] or match[2]))
            else:
                discarded.append(source)
    targets = [target for _, target, _ in candidates]
    if len(set(targets)) != len(targets) or any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError(f"Duplicate/existing dump names in {destination}; no files removed")
    destination.mkdir(parents=True, exist_ok=True)
    retained = []
    for source, target, kind in candidates:
        size = source.stat().st_size
        original = str(source.relative_to(run))
        source.rename(target)
        retained.append(dict(kind=kind, original=original, file=target.name, bytes=size))
    removed_bytes = 0
    if not keep_all:
        for path in discarded:
            removed_bytes += path.stat().st_size
            path.unlink()
    present = {item["kind"] for item in retained if item["bytes"] > 0}
    return dict(directory=os.path.relpath(destination, run), files=retained,
                missing_or_empty=[k for k in KINDS if k not in present],
                removed_files=0 if keep_all else len(discarded), removed_bytes=removed_bytes,
                keep_all_dumps=keep_all)
