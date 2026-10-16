"""Untimed transport and readiness helpers for the Docker benchmark."""
from pathlib import Path
import subprocess


def collect_export(path: Path, container_path: str, engine: str) -> None:
    if engine != 'og':
        return
    result = subprocess.run(
        ['docker', 'exec', '-u', 'omm', 'bench-og', 'cat', '--', container_path],
        text=True, capture_output=True, timeout=180, check=True)
    # Only generated public benchmark data are republished. No database
    # directory permissions or server security settings are changed.
    path.unlink()
    path.write_text(result.stdout)


def require_final_server(engine: str) -> None:
    result = subprocess.run(
        ['docker', 'exec', 'bench-' + engine, 'cat', '/proc/1/comm'],
        text=True, capture_output=True, timeout=10, check=True)
    expected = 'postgres' if engine == 'pg' else 'gaussdb'
    if result.stdout.strip() != expected:
        raise RuntimeError(f'{engine} container is still in bootstrap: {result.stdout.strip()}')


if __name__ == '__main__':
    # Backward-compatible entrypoint used by the original measured workflows.
    import run
    run.main()
