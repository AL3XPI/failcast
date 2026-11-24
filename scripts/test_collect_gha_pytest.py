#!/usr/bin/env python3
"""Offline unit tests for scripts/collect_gha_pytest.py."""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_gha_pytest as collector  # noqa: E402


JUNIT_FIXTURE_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4" time="1.850">
    <testcase classname="tests.test_client" name="test_get_ok" time="0.125" />
    <testcase classname="tests.test_client.TestRetry" name="test_retry_backoff[3]" time="0.480">
      <failure message="AssertionError: expected 3 retries, got 2">Traceback...</failure>
    </testcase>
    <testcase classname="tests.test_auth" name="test_setup_error" time="0.015">
      <error message="RuntimeError: missing fixture">Fixture error</error>
    </testcase>
    <testcase classname="tests.test_tls" name="test_windows_only" time="0.000">
      <skipped type="pytest.skip" message="Windows only" />
    </testcase>
    <testcase classname="tests.ApplicationOAuth.ApplicationOAuth" name="testCheckError" time="0.022" />
  </testsuite>
</testsuites>
"""

VERBOSE_PYTEST_LOG_FIXTURE = """\
2026-01-12T10:00:01.1234567Z \x1b[1m============================= test session starts ==============================\x1b[0m
2026-01-12T10:00:01.2000000Z collected 5 items
2026-01-12T10:00:01.3000000Z tests/test_urls.py::test_parse_basic \x1b[32mPASSED\x1b[0m [ 20%]
2026-01-12T10:00:01.4000000Z tests/test_urls.py::TestQuery::test_escape[utf8] \x1b[31mFAILED\x1b[0m [ 40%]
2026-01-12T10:00:01.5000000Z [gw0] [ 60%] ERROR tests/test_stream.py::test_timeout_cleanup
2026-01-12T10:00:01.6000000Z tests/test_tls.py::test_legacy_cipher SKIPPED (OpenSSL 3 required) [ 80%]
2026-01-12T10:00:01.7000000Z [gw1] PASSED tests/test_urls.py::test_join_relative
2026-01-12T10:00:01.8000000Z ============================= slowest 5 durations ==============================
2026-01-12T10:00:01.8100000Z 0.35s call     tests/test_urls.py::TestQuery::test_escape[utf8]
2026-01-12T10:00:01.8150000Z 0.05s setup    tests/test_urls.py::TestQuery::test_escape[utf8]
2026-01-12T10:00:01.8200000Z 0.12s call     tests/test_urls.py::test_parse_basic
2026-01-12T10:00:01.9000000Z =========================== short test summary info ============================
2026-01-12T10:00:01.9100000Z FAILED tests/test_urls.py::TestQuery::test_escape[utf8] - AssertionError: bad percent encoding
2026-01-12T10:00:01.9200000Z ERROR tests/test_stream.py::test_timeout_cleanup - RuntimeError: loop closed
2026-01-12T10:00:02.0000000Z =================== 1 failed, 2 passed, 1 skipped, 1 error in 2.00s ===================
"""


class TestCollectGhaPytest(unittest.TestCase):
    def test_parse_junit_xml_artifact_zip(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("reports/junit.xml", JUNIT_FIXTURE_XML)

        parsed = collector.parse_junit_zip_bytes(buf.getvalue())
        self.assertEqual(parsed.source, "junit_xml")
        self.assertEqual(len(parsed.outcomes), 5)
        self.assertEqual(parsed.exact_duration_count, 5)

        by_id = {o["test_id"]: o for o in parsed.outcomes}
        self.assertIn("tests/test_client.py::test_get_ok", by_id)
        self.assertEqual(by_id["tests/test_client.py::test_get_ok"]["status"], "pass")
        self.assertEqual(by_id["tests/test_client.py::test_get_ok"]["duration_ms"], 125)
        self.assertEqual(by_id["tests/test_client.py::test_get_ok"]["file_path"], "tests/test_client.py")

        retry = by_id["tests/test_client.py::TestRetry::test_retry_backoff[3]"]
        self.assertEqual(retry["status"], "fail")
        self.assertEqual(retry["duration_ms"], 480)
        self.assertIn("expected 3 retries", retry["failure_excerpt"])

        err = by_id["tests/test_auth.py::test_setup_error"]
        self.assertEqual(err["status"], "error")
        self.assertEqual(err["duration_ms"], 15)

        skipped = by_id["tests/test_tls.py::test_windows_only"]
        self.assertEqual(skipped["status"], "skip")
        self.assertEqual(skipped["duration_ms"], 0)

        camel = by_id["tests/ApplicationOAuth.py::ApplicationOAuth::testCheckError"]
        self.assertEqual(camel["status"], "pass")
        self.assertEqual(camel["file_path"], "tests/ApplicationOAuth.py")
        self.assertEqual(camel["duration_ms"], 22)

    def test_parse_verbose_pytest_log_with_durations_and_fallback(self) -> None:
        parsed = collector.parse_pytest_log(VERBOSE_PYTEST_LOG_FIXTURE)
        self.assertEqual(parsed.source, "pytest_log")
        self.assertEqual(len(parsed.outcomes), 5)
        # Two tests had explicit --durations entries (test_escape + test_parse_basic).
        self.assertEqual(parsed.exact_duration_count, 2)
        # 2.00s across 4 non-skip tests => 500ms fallback estimate.
        self.assertEqual(parsed.fallback_duration_ms, 500)

        by_id = {o["test_id"]: o for o in parsed.outcomes}

        escape = by_id["tests/test_urls.py::TestQuery::test_escape[utf8]"]
        self.assertEqual(escape["status"], "fail")
        # 0.35s call + 0.05s setup = 400 ms exact.
        self.assertEqual(escape["duration_ms"], 400)
        self.assertEqual(escape["file_path"], "tests/test_urls.py")
        self.assertEqual(escape["failure_excerpt"], "AssertionError: bad percent encoding")

        basic = by_id["tests/test_urls.py::test_parse_basic"]
        self.assertEqual(basic["status"], "pass")
        self.assertEqual(basic["duration_ms"], 120)

        err = by_id["tests/test_stream.py::test_timeout_cleanup"]
        self.assertEqual(err["status"], "error")
        self.assertEqual(err["duration_ms"], 500)
        self.assertEqual(err["failure_excerpt"], "RuntimeError: loop closed")

        skipped = by_id["tests/test_tls.py::test_legacy_cipher"]
        self.assertEqual(skipped["status"], "skip")
        self.assertEqual(skipped["duration_ms"], 0)

        join_rel = by_id["tests/test_urls.py::test_join_relative"]
        self.assertEqual(join_rel["status"], "pass")
        self.assertEqual(join_rel["duration_ms"], 500)

    def test_parse_iso_timestamp_chronological_order(self) -> None:
        t1 = collector.parse_iso_timestamp("2025-11-01T08:00:00Z")
        t2 = collector.parse_iso_timestamp("2025-11-01T09:30:00Z")
        self.assertLess(t1, t2)
        self.assertEqual(t2 - t1, 5400)

    def test_non_verbose_summary_only_log_is_rejected(self) -> None:
        summary_only = """\
=========================== short test summary info ============================
FAILED tests/test_urls.py::test_a - AssertionError: boom
FAILED tests/test_urls.py::test_b - AssertionError: boom
=================== 2 failed, 100 passed in 3.10s ===================
"""
        parsed = collector.parse_pytest_log(summary_only)
        self.assertEqual(parsed.outcomes, [])


if __name__ == "__main__":
    unittest.main()
