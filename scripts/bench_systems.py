#!/usr/bin/env python3
"""Reproducible systems latency and binary footprint benchmark for `pts`.

Builds `target/release/pts` and `pts-ml`'s `bench_infer` example, then measures:
  - Stripped release binary size (MB)
  - Median and p95 cold-start latency (15 runs) of `pts run --budget 15 --dry-run --json`
  - Median and p95 latency (15 runs) of `pts explain <test_id> --json`
  - Pure `pts-ml` GBDT+Platt scoring throughput (ms per 1,000 tests and tests/sec)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


def run_cmd(
    cmd: list[str],
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command {' '.join(cmd)} failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def detect_cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        proc = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def detect_rustc_version() -> str:
    proc = run_cmd(["rustc", "--version"], check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def measure_command_latencies(
    cmd: list[str],
    cwd: Path,
    runs: int,
    warmup: int = 2,
) -> tuple[list[float], str]:
    last_stdout = ""
    for _ in range(warmup):
        res = run_cmd(cmd, cwd=cwd)
        last_stdout = res.stdout

    timings_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res = run_cmd(cmd, cwd=cwd)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(elapsed_ms)
        last_stdout = res.stdout
    return timings_ms, last_stdout


def measure_stripped_binary_mb(binary_path: Path) -> tuple[float, float]:
    raw_bytes = binary_path.stat().st_size
    with tempfile.TemporaryDirectory(prefix="pts-strip-") as tmpdir:
        copied = Path(tmpdir) / binary_path.name
        shutil.copy2(binary_path, copied)
        strip_bin = shutil.which("strip")
        if strip_bin:
            run_cmd([strip_bin, str(copied)], check=False)
        stripped_bytes = copied.stat().st_size
    return raw_bytes / (1024 * 1024), stripped_bytes / (1024 * 1024)


def create_offline_fixture_repo(pts_bin: Path, num_tests: int = 1000) -> Path:
    """Create a 1,000-test Python git repository with trained PTS model for offline benchmarking."""
    repo_dir = Path(tempfile.mkdtemp(prefix="pts-bench-fixture-"))
    src_dir = repo_dir / "src"
    tests_dir = repo_dir / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    num_modules = 20
    tests_per_module = max(1, num_tests // num_modules)

    for m in range(num_modules):
        (src_dir / f"mod_{m}.py").write_text(
            f"def compute_{m}(x: int) -> int:\n    return x + {m}\n",
            encoding="utf-8",
        )
        lines = [f"from src.mod_{m} import compute_{m}\n\n"]
        for t in range(tests_per_module):
            lines.append(
                f"def test_case_{m}_{t}():\n    assert compute_{m}({t}) == {t + m}\n\n"
            )
        (tests_dir / f"test_mod_{m}.py").write_text("".join(lines), encoding="utf-8")

    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / ".gitignore").write_text(".pts/\n__pycache__/\n", encoding="utf-8")

    run_cmd(["git", "init", "-q"], cwd=repo_dir)
    run_cmd(["git", "config", "user.name", "PTS Bench"], cwd=repo_dir)
    run_cmd(["git", "config", "user.email", "bench@example.com"], cwd=repo_dir)
    run_cmd(["git", "add", "."], cwd=repo_dir)
    run_cmd(["git", "commit", "-q", "-m", "Initial 1000-test fixture"], cwd=repo_dir)

    run_cmd([str(pts_bin), "init", "--history-limit", "50"], cwd=repo_dir)

    jsonl_path = repo_dir / ".pts" / "fixture_runs.jsonl"
    base_ts = 1_720_000_000
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in range(40):
            touched = r % num_modules
            outcomes = []
            for m in range(num_modules):
                fp = f"tests/test_mod_{m}.py"
                for t in range(tests_per_module):
                    tid = f"{fp}::test_case_{m}_{t}"
                    fail = m == touched and t == 0 and (r % 2 == 0)
                    outcomes.append(
                        {
                            "test_id": tid,
                            "file_path": fp,
                            "status": "fail" if fail else "pass",
                            "duration_ms": 20 + ((m * 7 + t) % 80),
                        }
                    )
            rec = {
                "run_id": f"fixture-{r:03d}",
                "started_at": base_ts + r * 3600,
                "diff_files": [
                    {"path": f"src/mod_{touched}.py", "lines_added": 12, "lines_deleted": 3}
                ],
                "outcomes": outcomes,
            }
            fh.write(json.dumps(rec) + "\n")

    run_cmd([str(pts_bin), "import-runs", str(jsonl_path)], cwd=repo_dir)
    jsonl_path.unlink(missing_ok=True)
    run_cmd([str(pts_bin), "train"], cwd=repo_dir)

    # Touch one source file so dry-run & explain evaluate a non-empty working-tree diff.
    with (src_dir / "mod_0.py").open("a", encoding="utf-8") as fh:
        fh.write("\n# bench diff change\n")

    return repo_dir


def detect_repo_label(repo_dir: Path) -> str:
    proc = run_cmd(["git", "remote", "get-url", "origin"], cwd=repo_dir, check=False)
    if proc.returncode == 0:
        url = proc.stdout.strip().removesuffix(".git")
        if "github.com/" in url:
            return url.split("github.com/", 1)[1]
        if "github.com:" in url:
            return url.split("github.com:", 1)[1]
    return repo_dir.name


def pick_tracked_python_file(repo_dir: Path) -> Optional[Path]:
    proc = run_cmd(["git", "ls-files", "*.py"], cwd=repo_dir, check=False)
    if proc.returncode != 0:
        return None
    candidates = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith("tests/") and not line.startswith("test_")
    ]
    if not candidates:
        candidates = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for rel in candidates:
        p = repo_dir / rel
        if p.is_file():
            return p
    return None


def render_markdown_table(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    speedup = (
        summary["cold_dry_run_p50_ms"] / summary["warm_dry_run_p50_ms"]
        if summary["warm_dry_run_p50_ms"] > 0
        else 1.0
    )
    lines = [
        "### Systems Latency & Footprint Benchmarks",
        "",
        f"Measured on **{env['cpu']}** ({env['os']} {env['arch']}, `{env['rustc']}`), "
        f"target repo `{summary['target_repo']}` ({summary['discovered_tests']:,} discovered tests, "
        f"model mode: `{summary['model_mode']}`).",
        "",
        "| Metric | Value | Notes |",
        "| :--- | ---: | :--- |",
        f"| Stripped release binary size (`pts`) | **{summary['stripped_binary_mb']:.2f} MB** | Unstripped: {summary['raw_binary_mb']:.2f} MB (single static SQLite + GBDT + Tree-sitter + MCP binary) |",
        f"| Warm CLI `pts run --budget 15 --dry-run --json` (p50 / p95) | **{summary['warm_dry_run_p50_ms']:.1f} ms / {summary['warm_dry_run_p95_ms']:.1f} ms** | Pure Rust cached discovery ({speedup:.1f}x faster than cold; git diff + SQLite + AST + GBDT+Platt) |",
        f"| Warm CLI `pts explain <test_id> --json` (p50 / p95) | **{summary['warm_explain_p50_ms']:.1f} ms / {summary['warm_explain_p95_ms']:.1f} ms** | Pure Rust cached discovery over {summary['runs']} runs (16 features + signals + Platt probability) |",
        f"| Cold CLI `pts run --budget 15 --dry-run --refresh-tests --json` (p50 / p95) | **{summary['cold_dry_run_p50_ms']:.1f} ms / {summary['cold_dry_run_p95_ms']:.1f} ms** | Includes external Python + `pytest --collect-only` startup across {summary['discovered_tests']:,} tests |",
        f"| Cold CLI `pts explain <test_id> --refresh-tests --json` (p50 / p95) | **{summary['cold_explain_p50_ms']:.1f} ms / {summary['cold_explain_p95_ms']:.1f} ms** | Includes external Python + `pytest --collect-only` startup |",
        f"| Pure `pts-ml` scoring latency (1,000 tests, p50 / p95) | **{summary['ml_infer_p50_ms_per_1000']:.2f} ms / {summary['ml_infer_p95_ms_per_1000']:.2f} ms** | **{summary['ml_infer_tests_per_sec']:,.0f} tests/sec** (150-tree GBDT + Platt calibration) |",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible systems latency benchmarks for pts.")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=None,
        help="Path to an initialized Python repo (with .pts/store.db). If omitted, creates a 1,000-test offline fixture.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=15,
        help="Number of measured runs per CLI latency benchmark (default: 15).",
    )
    parser.add_argument(
        "--test-id",
        default=None,
        help="Specific test_id for `pts explain` benchmark (auto-detected if omitted).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark summary.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    print("[bench] Building release `pts` binary and `pts-ml` inference benchmark...", file=sys.stderr)
    run_cmd(["cargo", "build", "--release", "--bin", "pts"], cwd=workspace_root)
    run_cmd(
        ["cargo", "build", "--release", "--example", "bench_infer", "-p", "pts-ml"],
        cwd=workspace_root,
    )

    pts_bin = workspace_root / "target" / "release" / "pts"
    infer_bin = workspace_root / "target" / "release" / "examples" / "bench_infer"

    raw_mb, stripped_mb = measure_stripped_binary_mb(pts_bin)
    print(f"[bench] Binary size: stripped={stripped_mb:.2f} MB (raw={raw_mb:.2f} MB)", file=sys.stderr)

    infer_proc = run_cmd([str(infer_bin)], cwd=workspace_root)
    infer_data = json.loads(infer_proc.stdout.strip())
    print(
        f"[bench] pts-ml inference: {infer_data['median_ms_per_1000']:.3f} ms / 1,000 tests "
        f"({infer_data['tests_per_sec']:,.0f} tests/sec)",
        file=sys.stderr,
    )

    created_temp_repo = False
    touched_file: Optional[Path] = None
    original_content: Optional[bytes] = None

    if args.repo_dir is not None and (args.repo_dir / ".pts" / "store.db").is_file():
        repo_dir = args.repo_dir.resolve()
        target_label = detect_repo_label(repo_dir)
        if not (repo_dir / ".pts" / "model.json").is_file():
            run_cmd([str(pts_bin), "train"], cwd=repo_dir, check=False)
        # Ensure working tree has a small diff during measurement, then restore cleanly.
        diff_check = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir, check=False)
        if not diff_check.stdout.strip():
            candidate = pick_tracked_python_file(repo_dir)
            if candidate is not None:
                original_content = candidate.read_bytes()
                touched_file = candidate
                candidate.write_bytes(original_content + b"\n# pts systems benchmark temporary edit\n")
    else:
        print("[bench] Creating 1,000-test offline Python fixture repository...", file=sys.stderr)
        repo_dir = create_offline_fixture_repo(pts_bin, num_tests=1000)
        created_temp_repo = True
        target_label = "synthetic-1000-tests-fixture"

    try:
        print("[bench] Priming SQLite discovery cache...", file=sys.stderr)
        prime = run_cmd(
            [str(pts_bin), "run", "--budget", "15", "--dry-run", "--json", "--refresh-tests"],
            cwd=repo_dir,
        )
        dry_run_report = json.loads(prime.stdout)

        test_id = args.test_id
        if not test_id:
            selected = dry_run_report.get("selected") or []
            if selected:
                test_id = selected[0]["test_id"]
            else:
                raise RuntimeError("No selected tests returned by dry-run; pass --test-id explicitly.")

        print(
            f"[bench] Measuring WARM `pts run --budget 15 --dry-run --json` ({args.runs} runs, cached discovery)...",
            file=sys.stderr,
        )
        warm_dry_run_ms, dry_run_stdout = measure_command_latencies(
            [str(pts_bin), "run", "--budget", "15", "--dry-run", "--json"],
            cwd=repo_dir,
            runs=args.runs,
        )
        dry_run_report = json.loads(dry_run_stdout)

        print(
            f"[bench] Measuring WARM `pts explain {test_id} --json` ({args.runs} runs, cached discovery)...",
            file=sys.stderr,
        )
        warm_explain_ms, _ = measure_command_latencies(
            [str(pts_bin), "explain", test_id, "--json"],
            cwd=repo_dir,
            runs=args.runs,
        )

        print(
            f"[bench] Measuring COLD `pts run --budget 15 --dry-run --refresh-tests --json` ({args.runs} runs)...",
            file=sys.stderr,
        )
        cold_dry_run_ms, _ = measure_command_latencies(
            [str(pts_bin), "run", "--budget", "15", "--dry-run", "--json", "--refresh-tests"],
            cwd=repo_dir,
            runs=args.runs,
            warmup=1,
        )

        print(
            f"[bench] Measuring COLD `pts explain {test_id} --refresh-tests --json` ({args.runs} runs)...",
            file=sys.stderr,
        )
        cold_explain_ms, _ = measure_command_latencies(
            [str(pts_bin), "explain", test_id, "--json", "--refresh-tests"],
            cwd=repo_dir,
            runs=args.runs,
            warmup=1,
        )

        summary = {
            "environment": {
                "os": platform.system(),
                "os_release": platform.release(),
                "arch": platform.machine(),
                "cpu": detect_cpu_model(),
                "rustc": detect_rustc_version(),
            },
            "target_repo": target_label,
            "discovered_tests": int(dry_run_report.get("total_discovered_tests", 0)),
            "selected_tests": int(dry_run_report.get("selected_count", 0)),
            "model_mode": str(dry_run_report.get("model_mode", "unknown")),
            "runs": args.runs,
            "raw_binary_mb": round(raw_mb, 2),
            "stripped_binary_mb": round(stripped_mb, 2),
            "warm_dry_run_p50_ms": round(percentile(warm_dry_run_ms, 0.50), 2),
            "warm_dry_run_p95_ms": round(percentile(warm_dry_run_ms, 0.95), 2),
            "warm_explain_p50_ms": round(percentile(warm_explain_ms, 0.50), 2),
            "warm_explain_p95_ms": round(percentile(warm_explain_ms, 0.95), 2),
            "cold_dry_run_p50_ms": round(percentile(cold_dry_run_ms, 0.50), 2),
            "cold_dry_run_p95_ms": round(percentile(cold_dry_run_ms, 0.95), 2),
            "cold_explain_p50_ms": round(percentile(cold_explain_ms, 0.50), 2),
            "cold_explain_p95_ms": round(percentile(cold_explain_ms, 0.95), 2),
            "explain_test_id": test_id,
            "ml_infer_batch_size": int(infer_data["batch_size"]),
            "ml_infer_p50_ms_per_1000": float(infer_data["median_ms_per_1000"]),
            "ml_infer_p95_ms_per_1000": float(infer_data["p95_ms_per_1000"]),
            "ml_infer_tests_per_sec": float(infer_data["tests_per_sec"]),
        }

        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"[bench] Wrote JSON summary to {args.out_json}", file=sys.stderr)

        print(render_markdown_table(summary))
        print("\n```json")
        print(json.dumps(summary, indent=2))
        print("```")
        return 0
    finally:
        if touched_file is not None and original_content is not None:
            touched_file.write_bytes(original_content)
        if created_temp_repo:
            shutil.rmtree(repo_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
