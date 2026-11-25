#!/usr/bin/env python3
"""Collect real pre-merge GitHub Actions pytest PR runs into failcast JSONL format.

Uses only the Python standard library and the `gh` CLI (respecting existing `gh auth`
credentials and GitHub API rate limits).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
GHA_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s?")

# Standard pytest -v/-vv line:
#   tests/test_foo.py::test_bar PASSED [ 12%]
#   tests/test_foo.py::test_bar[a-b] FAILED [ 42%]
#   tests/test_foo.py::test_skip SKIPPED (reason) [ 50%]
PYTEST_VERBOSE_RE = re.compile(
    r"^(?P<nodeid>[^\s:]+\.py::\S+)\s+"
    r"(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r"(?:\s+\([^)]*\))?"
    r"(?:\s+\[\s*\d+%\])?\s*$"
)

# pytest-xdist (-n auto) verbose line:
#   [gw0] [ 12%] PASSED tests/test_foo.py::test_bar
#   [gw1] FAILED tests/test_foo.py::test_bar
PYTEST_XDIST_RE = re.compile(
    r"^\[gw\d+\]\s+(?:\[\s*\d+%\]\s+)?"
    r"(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+"
    r"(?P<nodeid>[^\s:]+\.py::\S+)"
    r"(?:\s+.*)?$"
)

# Short test summary info lines:
#   FAILED tests/test_foo.py::test_bar - AssertionError: ...
#   ERROR tests/test_foo.py::test_baz - RuntimeError: ...
PYTEST_SUMMARY_FAIL_RE = re.compile(
    r"^(?P<status>FAILED|ERROR)\s+(?P<nodeid>[^\s:]+\.py::\S+?)(?:\s+-\s+(?P<msg>.+))?$"
)

# pytest --durations=N lines:
#   1.23s call     tests/test_foo.py::test_slow
#   0.04s setup    tests/test_foo.py::test_slow
PYTEST_DURATION_RE = re.compile(
    r"^(?P<secs>\d+(?:\.\d+)?)s\s+(?:call|setup|teardown)\s+(?P<nodeid>[^\s:]+\.py::\S+)\s*$"
)

# pytest final summary duration line:
#   === 2 failed, 140 passed, 3 skipped in 14.52s ===
PYTEST_TOTAL_TIME_RE = re.compile(
    r"=+\s+.*?\bin\s+(?P<secs>\d+(?:\.\d+)?)s\b.*?=+"
)

STATUS_MAP = {
    "PASSED": "pass",
    "FAILED": "fail",
    "ERROR": "error",
    "SKIPPED": "skip",
    "XFAIL": "pass",
    "XPASS": "pass",
}

DEFAULT_FALLBACK_DURATION_MS = 50
MIN_RATE_LIMIT_REMAINING = 40
MAX_FAILURE_EXCERPT_CHARS = 1000


@dataclass
class ParsedOutcomes:
    outcomes: list[dict[str, Any]]
    exact_duration_count: int
    source: str
    fallback_duration_ms: Optional[int] = None


def normalize_posix_path(raw: str) -> str:
    path = raw.replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    while path.startswith("/"):
        path = path[1:]
    return path


def file_path_from_test_id(test_id: str) -> str:
    return normalize_posix_path(test_id.split("::", 1)[0])


def strip_log_prefix(line: str) -> str:
    cleaned = GHA_TIMESTAMP_RE.sub("", line.rstrip("\r\n"))
    cleaned = ANSI_ESCAPE_RE.sub("", cleaned)
    return cleaned.strip()


def parse_iso_timestamp(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def junit_testcase_to_test_id(classname: str, name: str, file_attr: str = "") -> tuple[str, str]:
    """Convert JUnit XML testcase attributes into pytest-style (test_id, file_path)."""
    clean_name = name.strip()
    if "::" in clean_name and ".py::" in clean_name:
        test_id = normalize_posix_path(clean_name)
        return test_id, file_path_from_test_id(test_id)

    file_path = normalize_posix_path(file_attr) if file_attr.strip() else ""
    class_parts: list[str] = []

    if classname.strip():
        parts = [p for p in classname.strip().split(".") if p]
        if file_path:
            mod_stem = file_path[:-3] if file_path.endswith(".py") else file_path
            mod_parts = [p for p in mod_stem.split("/") if p]
            if len(parts) > len(mod_parts) and parts[: len(mod_parts)] == mod_parts:
                class_parts = parts[len(mod_parts) :]
            elif parts and parts[-1][:1].isupper():
                class_parts = [parts[-1]]
        else:
            # Split dotted classname into module path and optional CamelCase class names.
            # Handle CamelCase test filenames (e.g. `tests.ApplicationOAuth.ApplicationOAuth`)
            # as well as snake_case modules with nested classes (`tests.test_client.TestRetry`).
            if len(parts) >= 2 and parts[-1] == parts[-2] and parts[-1][:1].isupper():
                mod_parts = parts[:-1]
                class_parts = [parts[-1]]
            else:
                split_idx = len(parts)
                while split_idx > 1 and parts[split_idx - 1][:1].isupper():
                    # Keep at least 2 path components when the root is a lowercase package
                    # like `tests` and the second component is a PascalCase test file.
                    if split_idx == 2 and len(parts) >= 3 and parts[0][:1].islower():
                        break
                    split_idx -= 1
                mod_parts = parts[:split_idx]
                class_parts = parts[split_idx:]
            if mod_parts:
                file_path = normalize_posix_path("/".join(mod_parts) + ".py")
            elif parts:
                file_path = normalize_posix_path(parts[0] + ".py")
    elif "." in clean_name and not clean_name.endswith(".py"):
        # pytest collection error testcase with empty classname and dotted module name.
        mod_path = normalize_posix_path("/".join(clean_name.split(".")) + ".py")
        return f"{mod_path}::<collection>", mod_path

    if not file_path:
        file_path = "tests/unknown.py"

    if class_parts:
        test_id = f"{file_path}::{'::'.join(class_parts)}::{clean_name}"
    else:
        test_id = f"{file_path}::{clean_name}"
    return test_id, file_path


def parse_junit_xml_bytes(xml_bytes: bytes) -> ParsedOutcomes:
    """Parse JUnit XML bytes into failcast outcome dicts with exact durations."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ParsedOutcomes(outcomes=[], exact_duration_count=0, source="junit_xml")

    outcomes_by_id: dict[str, dict[str, Any]] = {}
    exact_ids: set[str] = set()

    for tc in root.iter("testcase"):
        name = tc.attrib.get("name", "").strip()
        if not name:
            continue
        classname = tc.attrib.get("classname", "")
        file_attr = tc.attrib.get("file", "")
        test_id, file_path = junit_testcase_to_test_id(classname, name, file_attr)

        status = "pass"
        failure_excerpt: Optional[str] = None
        failure_el = tc.find("failure")
        error_el = tc.find("error")
        skipped_el = tc.find("skipped")

        if failure_el is not None:
            status = "fail"
            msg = failure_el.attrib.get("message") or (failure_el.text or "").strip()
            failure_excerpt = msg[:MAX_FAILURE_EXCERPT_CHARS] if msg else None
        elif error_el is not None:
            status = "error"
            msg = error_el.attrib.get("message") or (error_el.text or "").strip()
            failure_excerpt = msg[:MAX_FAILURE_EXCERPT_CHARS] if msg else None
        elif skipped_el is not None:
            status = "skip"

        time_raw = tc.attrib.get("time")
        duration_ms = DEFAULT_FALLBACK_DURATION_MS
        has_exact = False
        if time_raw is not None:
            try:
                secs = max(0.0, float(time_raw))
                duration_ms = 0 if status == "skip" else max(1, int(round(secs * 1000.0)))
                has_exact = True
            except ValueError:
                pass

        outcome: dict[str, Any] = {
            "test_id": test_id,
            "file_path": file_path,
            "status": status,
            "duration_ms": duration_ms,
        }
        if failure_excerpt:
            outcome["failure_excerpt"] = failure_excerpt

        # If duplicate test_id appears (e.g. reruns), prefer failure/error or latest exact result.
        existing = outcomes_by_id.get(test_id)
        if existing is None or status in ("fail", "error") or existing["status"] == "skip":
            outcomes_by_id[test_id] = outcome
            if has_exact:
                exact_ids.add(test_id)

    outcomes = list(outcomes_by_id.values())
    return ParsedOutcomes(
        outcomes=outcomes,
        exact_duration_count=len(exact_ids),
        source="junit_xml",
    )


def parse_junit_zip_bytes(zip_bytes: bytes) -> ParsedOutcomes:
    """Extract and parse all JUnit XML files from a GitHub Actions artifact zip."""
    merged: dict[str, dict[str, Any]] = {}
    exact_ids: set[str] = set()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".xml"):
                    continue
                if info.file_size > 30_000_000:
                    continue
                parsed = parse_junit_xml_bytes(zf.read(info.filename))
                for item in parsed.outcomes:
                    tid = item["test_id"]
                    prev = merged.get(tid)
                    if prev is None or item["status"] in ("fail", "error"):
                        merged[tid] = item
                    exact_ids.add(tid)
    except zipfile.BadZipFile:
        return ParsedOutcomes(outcomes=[], exact_duration_count=0, source="junit_xml")

    return ParsedOutcomes(
        outcomes=list(merged.values()),
        exact_duration_count=min(len(exact_ids), len(merged)),
        source="junit_xml",
    )


def parse_pytest_log(log_text: str) -> ParsedOutcomes:
    """Parse verbose pytest job logs (and optional --durations output) into failcast outcomes."""
    statuses: dict[str, str] = {}
    excerpts: dict[str, str] = {}
    durations_ms: dict[str, int] = {}
    total_suite_secs: Optional[float] = None
    verbose_line_count = 0

    for raw_line in log_text.splitlines():
        line = strip_log_prefix(raw_line)
        if not line:
            continue

        m = PYTEST_VERBOSE_RE.match(line) or PYTEST_XDIST_RE.match(line)
        if m:
            verbose_line_count += 1
            nodeid = normalize_posix_path(m.group("nodeid"))
            status = STATUS_MAP[m.group("status")]
            prev = statuses.get(nodeid)
            if prev is None or status in ("fail", "error") or prev == "skip":
                statuses[nodeid] = status
            continue

        dm = PYTEST_DURATION_RE.match(line)
        if dm:
            nodeid = normalize_posix_path(dm.group("nodeid"))
            secs = max(0.0, float(dm.group("secs")))
            ms = max(1, int(round(secs * 1000.0))) if secs > 0.0 else 1
            durations_ms[nodeid] = durations_ms.get(nodeid, 0) + ms
            continue

        sm = PYTEST_SUMMARY_FAIL_RE.match(line)
        if sm:
            nodeid = normalize_posix_path(sm.group("nodeid"))
            status = STATUS_MAP[sm.group("status")]
            statuses[nodeid] = status
            msg = sm.group("msg")
            if msg and nodeid not in excerpts:
                excerpts[nodeid] = msg.strip()[:MAX_FAILURE_EXCERPT_CHARS]
            continue

        tm = PYTEST_TOTAL_TIME_RE.search(line)
        if tm:
            try:
                total_suite_secs = float(tm.group("secs"))
            except ValueError:
                pass

    # If pytest was not executed in verbose mode, summary-only scraping would omit all
    # passing tests and distort the dataset; return empty so caller skips this job.
    if not statuses or verbose_line_count == 0:
        return ParsedOutcomes(outcomes=[], exact_duration_count=0, source="pytest_log")

    non_skip_count = sum(1 for st in statuses.values() if st != "skip")
    if total_suite_secs is not None and total_suite_secs > 0 and non_skip_count > 0:
        est_ms = int(round((total_suite_secs * 1000.0) / non_skip_count))
        fallback_ms = max(5, min(est_ms, 5000))
    else:
        fallback_ms = DEFAULT_FALLBACK_DURATION_MS

    outcomes: list[dict[str, Any]] = []
    exact_count = 0
    for nodeid, status in statuses.items():
        if nodeid in durations_ms:
            dur = durations_ms[nodeid]
            exact_count += 1
        elif status == "skip":
            dur = 0
        else:
            dur = fallback_ms

        entry: dict[str, Any] = {
            "test_id": nodeid,
            "file_path": file_path_from_test_id(nodeid),
            "status": status,
            "duration_ms": dur,
        }
        if nodeid in excerpts:
            entry["failure_excerpt"] = excerpts[nodeid]
        outcomes.append(entry)

    return ParsedOutcomes(
        outcomes=outcomes,
        exact_duration_count=exact_count,
        source="pytest_log",
        fallback_duration_ms=fallback_ms if exact_count < len(outcomes) else None,
    )


class GhApiClient:
    """Thin wrapper around `gh api` respecting existing `gh auth` and rate limits."""

    def __init__(self, repo: str, verbose: bool = False) -> None:
        self.repo = repo
        self.verbose = verbose

    def _run_gh(
        self,
        args: list[str],
        binary: bool = False,
        allow_escape_sequences: bool = False,
    ) -> bytes:
        cmd = ["gh", "api"]
        if allow_escape_sequences:
            cmd.append("--allow-escape-sequences")
        cmd.extend(args)
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 and allow_escape_sequences:
            err = proc.stderr.decode("utf-8", errors="replace")
            if "unknown flag" in err.lower():
                proc = subprocess.run(["gh", "api", *args], capture_output=True, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"gh api {' '.join(args)} failed ({proc.returncode}): {err}")
        return proc.stdout

    def get_json(self, endpoint: str) -> Any:
        raw = self._run_gh([endpoint])
        return json.loads(raw.decode("utf-8"))

    def get_bytes(self, endpoint: str, allow_escape_sequences: bool = False) -> bytes:
        return self._run_gh(
            [endpoint],
            binary=True,
            allow_escape_sequences=allow_escape_sequences,
        )

    def check_rate_limit(self) -> tuple[int, int]:
        data = self.get_json("rate_limit")
        core = data.get("resources", {}).get("core", {})
        remaining = int(core.get("remaining", 0))
        limit = int(core.get("limit", 0))
        return remaining, limit

    def resolve_workflow_id(self, workflow_filter: str) -> Optional[int]:
        data = self.get_json(f"repos/{self.repo}/actions/workflows?per_page=100")
        needle = workflow_filter.strip().lower()
        for wf in data.get("workflows", []):
            name = str(wf.get("name", "")).lower()
            path = str(wf.get("path", "")).lower()
            if needle == name or needle in name or needle in path:
                return int(wf["id"])
        return None

    def list_pr_runs(
        self,
        workflow_id: Optional[int],
        max_pages: int = 12,
    ) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        base = (
            f"repos/{self.repo}/actions/workflows/{workflow_id}/runs"
            if workflow_id is not None
            else f"repos/{self.repo}/actions/runs"
        )
        now_ts = int(datetime.now(timezone.utc).timestamp())
        for page in range(1, max_pages + 1):
            endpoint = f"{base}?event=pull_request&status=completed&per_page=100&page={page}"
            data = self.get_json(endpoint)
            page_runs = data.get("workflow_runs", [])
            if not page_runs:
                break
            fresh = [
                r
                for r in page_runs
                if (r.get("run_started_at") or r.get("created_at"))
                and now_ts - parse_iso_timestamp(r.get("run_started_at") or r["created_at"]) <= 89 * 86400
            ]
            runs.extend(fresh)
            if len(fresh) < len(page_runs) or len(page_runs) < 100:
                break
        return runs

    def fetch_diff_files(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        pr_number: Optional[int] = None
        prs = run.get("pull_requests") or []
        if prs and isinstance(prs[0], dict) and prs[0].get("number"):
            pr_number = int(prs[0]["number"])

        head_sha = run.get("head_sha")
        if pr_number is None and head_sha:
            try:
                commit_prs = self.get_json(f"repos/{self.repo}/commits/{head_sha}/pulls")
                if isinstance(commit_prs, list) and commit_prs:
                    pr_number = int(commit_prs[0]["number"])
            except RuntimeError:
                pr_number = None

        files_raw: list[dict[str, Any]] = []
        if pr_number is not None:
            try:
                files_raw = self.get_json(
                    f"repos/{self.repo}/pulls/{pr_number}/files?per_page=100"
                )
            except RuntimeError:
                files_raw = []

        if not files_raw and head_sha:
            try:
                commit_data = self.get_json(f"repos/{self.repo}/commits/{head_sha}")
                files_raw = commit_data.get("files", [])
            except RuntimeError:
                files_raw = []

        diff_files: list[dict[str, Any]] = []
        for f in files_raw:
            fname = f.get("filename")
            if not fname:
                continue
            diff_files.append(
                {
                    "path": normalize_posix_path(str(fname)),
                    "lines_added": int(f.get("additions", 0)),
                    "lines_deleted": int(f.get("deletions", 0)),
                }
            )
        return diff_files

    def list_jobs(self, run_id: int) -> list[dict[str, Any]]:
        try:
            jobs_data = self.get_json(f"repos/{self.repo}/actions/runs/{run_id}/jobs?per_page=100")
            return jobs_data.get("jobs", [])
        except RuntimeError:
            return []

    def fetch_junit_outcomes(
        self,
        run_id: int,
        prefer_failed: bool = False,
        failed_job_names: Optional[list[str]] = None,
    ) -> Optional[ParsedOutcomes]:
        try:
            data = self.get_json(f"repos/{self.repo}/actions/runs/{run_id}/artifacts")
        except RuntimeError:
            return None

        artifacts = data.get("artifacts", [])
        failed_tokens: set[str] = set()
        for jn in failed_job_names or []:
            for tok in re.findall(r"[a-z0-9.]+", jn.lower()):
                if len(tok) >= 3:
                    failed_tokens.add(tok)

        candidates = []
        for art in artifacts:
            if art.get("expired"):
                continue
            name = str(art.get("name", "")).lower()
            size = int(art.get("size_in_bytes", 0))
            if size == 0 or size > 20_000_000:
                continue
            if any(skip in name for skip in ("coverage", "dist", "wheel", "sdist", "openapi")):
                continue
            score = 0
            if "junit" in name:
                score += 12
            if "test" in name or "pytest" in name or "result" in name or "xml" in name:
                score += 6
            if score == 0:
                continue
            if failed_tokens:
                overlap = sum(1 for tok in failed_tokens if tok in name)
                score += overlap * 5
            if "ubuntu" in name or "linux" in name:
                score += 4
            if any(v in name for v in ("3.11", "3.12", "3.13")):
                score += 2
            candidates.append((score, art))

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        best_parsed: Optional[ParsedOutcomes] = None
        max_try = 3 if prefer_failed else 1
        for _, art in candidates[:max_try]:
            art_id = art["id"]
            try:
                zip_bytes = self.get_bytes(
                    f"repos/{self.repo}/actions/artifacts/{art_id}/zip"
                )
            except RuntimeError:
                continue
            parsed = parse_junit_zip_bytes(zip_bytes)
            if not parsed.outcomes:
                continue
            has_fail = any(o["status"] in ("fail", "error") for o in parsed.outcomes)
            if prefer_failed and has_fail:
                return parsed
            if best_parsed is None or len(parsed.outcomes) > len(best_parsed.outcomes):
                best_parsed = parsed
                if not prefer_failed:
                    return best_parsed
        return best_parsed

    def fetch_pytest_log_outcomes(
        self,
        run_id: int,
        prefer_failed: bool,
        jobs: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[ParsedOutcomes]:
        if jobs is None:
            jobs = self.list_jobs(run_id)
        if not jobs:
            return None

        def is_non_test_job(name_lower: str) -> bool:
            return any(
                k in name_lower
                for k in (
                    "lint",
                    "ruff",
                    "mypy",
                    "flake",
                    "docs",
                    "typing",
                    "pre-commit",
                    "format",
                    "build",
                    "publish",
                    "style",
                    "coverage",
                    "codecov",
                    "sonar",
                    "label",
                )
            )

        def job_rank(job: dict[str, Any]) -> tuple[int, int, int]:
            name = str(job.get("name", "")).lower()
            conclusion = str(job.get("conclusion", "")).lower()
            is_version_job = 1 if re.search(r"\b3\.\d+", name) else 0
            is_pytest_like = (
                1
                if (
                    is_version_job
                    or any(k in name for k in ("test", "pytest", "python", "unit", "linux", "ubuntu"))
                )
                else 0
            )
            non_test = 1 if is_non_test_job(name) else 0
            is_stable_linux = (
                1
                if any(k in name for k in ("ubuntu", "linux", "3.11", "3.12", "3.13"))
                and "dev" not in name
                and "min" not in name
                else 0
            )
            fail_match = (
                1
                if (prefer_failed and conclusion == "failure")
                else (1 if (not prefer_failed and conclusion == "success") else 0)
            )
            return (fail_match, is_pytest_like - non_test, is_stable_linux)

        if prefer_failed:
            candidate_jobs = [
                j
                for j in jobs
                if str(j.get("conclusion", "")).lower() == "failure"
                and not is_non_test_job(str(j.get("name", "")).lower())
            ]
        else:
            candidate_jobs = [
                j
                for j in jobs
                if str(j.get("conclusion", "")).lower() == "success"
                and not is_non_test_job(str(j.get("name", "")).lower())
            ]
            if not candidate_jobs:
                candidate_jobs = [j for j in jobs if str(j.get("conclusion", "")).lower() == "success"]

        if not candidate_jobs:
            return None

        ranked_jobs = sorted(candidate_jobs, key=job_rank, reverse=True)
        best_parsed: Optional[ParsedOutcomes] = None

        for job in ranked_jobs[:3]:
            job_id = job.get("id")
            if not job_id:
                continue
            try:
                raw_log = self.get_bytes(
                    f"repos/{self.repo}/actions/jobs/{job_id}/logs",
                    allow_escape_sequences=True,
                )
            except RuntimeError:
                continue
            text = raw_log.decode("utf-8", errors="replace")
            parsed = parse_pytest_log(text)
            if len(parsed.outcomes) < 5:
                continue
            has_fail = any(o["status"] in ("fail", "error") for o in parsed.outcomes)
            if prefer_failed and has_fail:
                return parsed
            if best_parsed is None or len(parsed.outcomes) > len(best_parsed.outcomes):
                best_parsed = parsed
                if not prefer_failed:
                    return best_parsed

        return best_parsed


def select_balanced_candidates(
    all_runs: list[dict[str, Any]],
    max_runs: int,
    min_failures: int,
    workflow_filter: Optional[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition completed PR runs into failed and passed pools (deduped by head_sha)."""
    failed: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for run in all_runs:
        run_id = run.get("id")
        if run_id in seen_ids:
            continue
        seen_ids.add(run_id)

        if workflow_filter:
            wf_name = str(run.get("name", "")).lower()
            wf_path = str(run.get("path", "")).lower()
            needle = workflow_filter.strip().lower()
            if needle not in wf_name and needle not in wf_path:
                continue

        conclusion = str(run.get("conclusion", "")).lower()
        if conclusion == "failure":
            failed.append(run)
        elif conclusion == "success":
            passed.append(run)

    # Cap scan pools generously so we have spares for lint-only failures or expired logs.
    max_fail_pool = max(min_failures * 4, int(max_runs * 0.75))
    max_pass_pool = max(max_runs * 2, 60)
    return failed[:max_fail_pool], passed[:max_pass_pool]


def collect_runs(
    repo: str,
    out_path: Path,
    max_runs: int,
    workflow: Optional[str],
    min_failures: int,
    min_tests_per_run: int = 10,
) -> dict[str, Any]:
    client = GhApiClient(repo=repo)
    remaining, limit = client.check_rate_limit()
    print(f"[collect] GitHub API core rate limit: {remaining}/{limit} remaining", file=sys.stderr)
    if remaining < MIN_RATE_LIMIT_REMAINING:
        raise RuntimeError(
            f"GitHub API rate limit too low ({remaining} remaining); aborting cleanly."
        )

    workflow_id: Optional[int] = None
    if workflow:
        workflow_id = client.resolve_workflow_id(workflow)
        if workflow_id is not None:
            print(f"[collect] Matched workflow '{workflow}' -> id={workflow_id}", file=sys.stderr)
        else:
            print(
                f"[collect] Workflow '{workflow}' not found by exact listing; filtering runs by name.",
                file=sys.stderr,
            )

    raw_runs = client.list_pr_runs(workflow_id=workflow_id, max_pages=15)
    print(f"[collect] Fetched {len(raw_runs)} completed PR workflow runs from {repo}", file=sys.stderr)

    failed_pool, passed_pool = select_balanced_candidates(
        raw_runs, max_runs=max_runs, min_failures=min_failures, workflow_filter=workflow
    )
    print(
        f"[collect] Candidate pools: {len(failed_pool)} failed runs, {len(passed_pool)} passed runs",
        file=sys.stderr,
    )

    # Target a balanced dataset: prioritize capturing real pytest failures first (up to ~40% of max_runs
    # or at least min_failures), then fill the remainder with passed runs.
    target_failures = min(max_runs, max(min_failures, int(round(max_runs * 0.35))))
    collected_records: list[dict[str, Any]] = []
    failing_runs_written = 0
    total_outcomes = 0
    exact_duration_outcomes = 0
    fallback_logged = False
    unique_tests: set[str] = set()
    used_run_ids: set[int] = set()

    def process_run(run: dict[str, Any], expect_failure: bool) -> Optional[dict[str, Any]]:
        nonlocal fallback_logged
        run_id = int(run["id"])
        if run_id in used_run_ids:
            return None

        # Skip runs older than 89 days (GitHub Actions expires logs/artifacts after 90 days).
        started_iso = run.get("run_started_at") or run.get("created_at")
        if not started_iso:
            return None
        started_ts = parse_iso_timestamp(started_iso)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if now_ts - started_ts > 89 * 86400:
            return None

        jobs: Optional[list[dict[str, Any]]] = None
        failed_test_job_names: list[str] = []
        if expect_failure:
            jobs = client.list_jobs(run_id)
            non_test_markers = (
                "lint",
                "ruff",
                "mypy",
                "flake",
                "docs",
                "typing",
                "pre-commit",
                "format",
                "build",
                "publish",
                "style",
                "schema",
                "openapi",
                "coverage",
                "codecov",
                "label",
            )
            failed_test_job_names = [
                str(j.get("name", ""))
                for j in jobs
                if str(j.get("conclusion", "")).lower() == "failure"
                and not any(m in str(j.get("name", "")).lower() for m in non_test_markers)
            ]
            if not failed_test_job_names:
                return None

        parsed = client.fetch_junit_outcomes(
            run_id,
            prefer_failed=expect_failure,
            failed_job_names=failed_test_job_names,
        )
        has_junit_failure = parsed is not None and any(
            o["status"] in ("fail", "error") for o in parsed.outcomes
        )
        if parsed is None or len(parsed.outcomes) < min_tests_per_run or (expect_failure and not has_junit_failure):
            log_parsed = client.fetch_pytest_log_outcomes(
                run_id, prefer_failed=expect_failure, jobs=jobs
            )
            if log_parsed is not None and len(log_parsed.outcomes) >= min_tests_per_run:
                parsed = log_parsed
        if parsed is None or len(parsed.outcomes) < min_tests_per_run:
            return None

        has_pytest_failure = any(o["status"] in ("fail", "error") for o in parsed.outcomes)
        has_pytest_pass = any(o["status"] == "pass" for o in parsed.outcomes)
        if not has_pytest_pass:
            # Skip total collection/environment crashes with zero passing tests.
            return None
        if expect_failure and not has_pytest_failure:
            # Skip non-pytest CI failures (e.g., mypy/ruff/docs) while filling the failure quota.
            return None

        diff_files = client.fetch_diff_files(run)
        if not diff_files:
            return None

        if parsed.fallback_duration_ms is not None and not fallback_logged:
            print(
                f"[collect] Note: per-test timings omitted in pytest logs for run {run_id}; "
                f"falling back to suite-averaged estimate (~{parsed.fallback_duration_ms} ms/test).",
                file=sys.stderr,
                flush=True,
            )
            fallback_logged = True

        used_run_ids.add(run_id)
        return {
            "record": {
                "run_id": f"gha-{run_id}",
                "commit_sha": run.get("head_sha"),
                "started_at": started_ts,
                "runner": "pytest",
                "diff_files": diff_files,
                "outcomes": parsed.outcomes,
            },
            "has_failure": has_pytest_failure,
            "exact_durations": parsed.exact_duration_count,
            "source": parsed.source,
        }

    for idx, run in enumerate(failed_pool):
        if failing_runs_written >= target_failures or len(collected_records) >= max_runs:
            break
        if idx > 0 and idx % 20 == 0:
            rem, _ = client.check_rate_limit()
            if rem < MIN_RATE_LIMIT_REMAINING:
                print(
                    f"[collect] Stopping early: GitHub API rate limit low ({rem} remaining).",
                    file=sys.stderr,
                )
                break
        res = process_run(run, expect_failure=True)
        if res is None:
            continue
        rec = res["record"]
        collected_records.append(rec)
        if res["has_failure"]:
            failing_runs_written += 1
        total_outcomes += len(rec["outcomes"])
        exact_duration_outcomes += res["exact_durations"]
        for o in rec["outcomes"]:
            unique_tests.add(o["test_id"])
        print(
            f"[collect] +fail run {rec['run_id']} ({len(rec['outcomes'])} tests, "
            f"{len(rec['diff_files'])} diff files, src={res['source']}) "
            f"[{failing_runs_written}/{target_failures} failing]",
            file=sys.stderr,
            flush=True,
        )

    remaining_slots = max_runs - len(collected_records)
    if remaining_slots > 0 and passed_pool:
        stride = max(1, len(passed_pool) // (remaining_slots * 2)) if len(passed_pool) > remaining_slots * 2 else 1
        ordered_passed = passed_pool[::stride] + [
            r for i, r in enumerate(passed_pool) if i % stride != 0
        ]
        for idx, run in enumerate(ordered_passed):
            if len(collected_records) >= max_runs:
                break
            if idx > 0 and idx % 20 == 0:
                rem, _ = client.check_rate_limit()
                if rem < MIN_RATE_LIMIT_REMAINING:
                    print(
                        f"[collect] Stopping early: GitHub API rate limit low ({rem} remaining).",
                        file=sys.stderr,
                    )
                    break
            res = process_run(run, expect_failure=False)
            if res is None:
                continue
            rec = res["record"]
            collected_records.append(rec)
            if res["has_failure"]:
                failing_runs_written += 1
            total_outcomes += len(rec["outcomes"])
            exact_duration_outcomes += res["exact_durations"]
            for o in rec["outcomes"]:
                unique_tests.add(o["test_id"])
            if len(collected_records) % 10 == 0 or len(collected_records) == max_runs:
                print(
                    f"[collect] progress: {len(collected_records)}/{max_runs} runs "
                    f"({failing_runs_written} failing)",
                    file=sys.stderr,
                    flush=True,
                )

    collected_records.sort(key=lambda r: (r["started_at"], r["run_id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in collected_records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    exact_duration_pct = (
        round((exact_duration_outcomes / total_outcomes) * 100.0, 2)
        if total_outcomes > 0
        else 0.0
    )
    summary = {
        "repo": repo,
        "out": str(out_path),
        "runs_written": len(collected_records),
        "failing_runs": failing_runs_written,
        "unique_tests": len(unique_tests),
        "total_outcomes": total_outcomes,
        "exact_duration_pct": exact_duration_pct,
        "min_started_at": collected_records[0]["started_at"] if collected_records else None,
        "max_started_at": collected_records[-1]["started_at"] if collected_records else None,
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect GitHub Actions PR pytest runs into failcast JSONL format."
    )
    parser.add_argument("--repo", required=True, help="GitHub repository in <owner/name> format.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=250,
        help="Maximum total PR runs to write (default: 250).",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Optional workflow name or path filter (e.g. 'Test Suite').",
    )
    parser.add_argument(
        "--min-failures",
        type=int,
        default=20,
        help="Target minimum number of failing pytest runs to prioritize (default: 20).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.max_runs <= 0:
        parser.error("--max-runs must be > 0")
    if args.min_failures < 0:
        parser.error("--min-failures must be >= 0")

    collect_runs(
        repo=args.repo,
        out_path=args.out,
        max_runs=args.max_runs,
        workflow=args.workflow,
        min_failures=args.min_failures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
