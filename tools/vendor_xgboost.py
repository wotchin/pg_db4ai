#!/usr/bin/env python3
"""Maintainer-only source import. Normal extension builds never download code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'third_party' / 'xgboost'
LOCK = json.loads((ROOT / 'third_party' / 'xgboost.lock.json').read_text())
MANIFEST = ROOT / 'third_party' / 'xgboost.manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def inventory():
    return {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        saved = json.loads(MANIFEST.read_text())
        if saved['source'] != LOCK or saved['files'] != inventory():
            raise SystemExit('Vendored source differs from its lock/manifest')
        print('Verified', len(saved['files']), 'vendored source files')
        return
    if DEST.exists():
        raise SystemExit('Refusing to overwrite third_party/xgboost; remove it explicitly when upgrading')
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'source.tar.gz'
        with urllib.request.urlopen(LOCK['url'], timeout=120) as response, archive.open('wb') as out:
            shutil.copyfileobj(response, out)
        if sha(archive) != LOCK['sha256']:
            raise SystemExit('XGBoost source archive SHA-256 mismatch')
        stage = Path(tmp) / 'tree'
        stage.mkdir()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts:
                    raise SystemExit('Unsafe archive path: ' + member.name)
                if not member.isfile() and not member.isdir():
                    raise SystemExit('Unsupported archive entry: ' + member.name)
                target = stage.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open('wb') as out:
                        shutil.copyfileobj(source, out)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
        roots = [p.parent for p in stage.rglob('CMakeLists.txt') if (p.parent / 'include/xgboost/c_api.h').is_file()]
        if len(roots) != 1:
            raise SystemExit('Cannot identify XGBoost source root')
        source = roots[0]
        if not (source / 'dmlc-core/CMakeLists.txt').is_file():
            raise SystemExit('Source archive is missing dmlc-core; refusing a network-dependent build')
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(DEST))
    files = inventory()
    MANIFEST.write_text(json.dumps({'source': LOCK, 'files': files}, indent=2, sort_keys=True) + '\n')
    print('Vendored', len(files), 'files from XGBoost', LOCK['version'])


if __name__ == '__main__':
    main()
