"""Untimed transport/readiness guards for the Docker benchmark.

openGauss COPY files belong to omm with mode 0600. Its image also starts a
short-lived bootstrap server before the final server. Neither behavior should
be misclassified as a failed ML algorithm.
"""
from pathlib import Path
import subprocess


def collect_export(path: Path, container_path: str, engine: str) -> None:
    if engine != 'og':
        return
    result = subprocess.run(
        ['docker', 'exec', '-u', 'omm', 'bench-og', 'cat', '--', container_path],
        text=True, capture_output=True, timeout=180, check=True)
    # Only generated, public benchmark data are republished. No database
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


def install() -> None:
    """Guard the harness transport; never retry timed training/prediction SQL."""
    import run
    original = run.sql

    def guarded_sql(engine, text, out=None, timeout=300):
        # run.start_servers already has a bounded readiness polling loop. A
        # successful connection to an entrypoint's temporary server is not ready.
        if out is None and text == 'SELECT version();':
            require_final_server(engine)
        result = original(engine, text, out, timeout)
        if engine == 'og' and out is not None and out.name.endswith('-prediction-export.log'):
            target = out.with_name(out.name.replace('-prediction-export.log', '-predictions.csv'))
            collect_export(target, '/bench/' + out.parent.name + '/' + target.name, engine)
        return result
    run.sql = guarded_sql


if __name__ == '__main__':
    install()
    import run
    run.main()
