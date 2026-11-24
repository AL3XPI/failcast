pub mod junit;
pub mod pytest;

pub use junit::{canonical_test_id, normalize_posix_path, parse_junit_xml, truncate_chars};
pub use pytest::PytestRunner;

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    use pts_core::{TestRunner, TestStatus};

    use super::*;

    fn mini_py_repo_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/mini_py_repo")
    }

    #[test]
    fn junit_xml_parses_pass_fail_error_skip_and_truncates() {
        let long_tb = "Traceback line: ".to_string() + &"X".repeat(5000);
        let xml = format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
  <testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4" time="1.805">
    <testcase classname="tests.test_auth" name="test_pass" file="tests/test_auth.py" time="0.125" />
    <testcase classname="tests.test_auth" name="test_fail_long" file="tests/test_auth.py" time="0.456">
      <failure message="AssertionError: bad token">{long_tb}</failure>
    </testcase>
    <testcase classname="tests.test_auth.TestSetup" name="test_error_case" time="1.200">
      <error message="RuntimeError: fixture boom">RuntimeError: fixture boom</error>
    </testcase>
    <testcase classname="tests.test_auth" name="test_skipped" time="0.002">
      <skipped type="pytest.skip" message="not ready">skipped</skipped>
    </testcase>
  </testsuite>
</testsuites>"#
        );

        let max_chars = 1500;
        let outcomes = parse_junit_xml(&xml, max_chars).expect("parse junit xml");
        assert_eq!(outcomes.len(), 4);

        assert_eq!(outcomes[0].test_id, "tests/test_auth.py::test_pass");
        assert_eq!(outcomes[0].status, TestStatus::Pass);
        assert_eq!(outcomes[0].duration_ms, 125);
        assert!(outcomes[0].failure_message.is_none());

        assert_eq!(outcomes[1].test_id, "tests/test_auth.py::test_fail_long");
        assert_eq!(outcomes[1].status, TestStatus::Fail);
        assert_eq!(outcomes[1].duration_ms, 456);
        let fail_msg = outcomes[1]
            .failure_message
            .as_ref()
            .expect("failure message");
        assert!(fail_msg.chars().count() <= max_chars);
        assert_eq!(fail_msg.chars().count(), max_chars);
        assert!(fail_msg.contains("AssertionError: bad token"));

        assert_eq!(
            outcomes[2].test_id,
            "tests/test_auth.py::TestSetup::test_error_case"
        );
        assert_eq!(outcomes[2].status, TestStatus::Error);
        assert_eq!(outcomes[2].duration_ms, 1200);
        assert_eq!(
            outcomes[2].failure_message.as_deref(),
            Some("RuntimeError: fixture boom")
        );

        assert_eq!(outcomes[3].test_id, "tests/test_auth.py::test_skipped");
        assert_eq!(outcomes[3].status, TestStatus::Skip);
        assert_eq!(outcomes[3].duration_ms, 2);
        assert!(outcomes[3].failure_message.is_none());
    }

    #[test]
    fn pytest_runner_discover_and_execute_fixture() {
        let fixture = mini_py_repo_path();
        assert!(
            fixture.is_dir(),
            "fixture repo missing at {}",
            fixture.display()
        );

        let runner = PytestRunner::default();
        let discovered = runner.discover(&fixture).expect("discover tests");
        let ids: Vec<&str> = discovered.iter().map(|t| t.test_id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "tests/test_calc.py::test_add_positive",
                "tests/test_calc.py::test_add_negative",
                "tests/test_calc.py::test_divide_broken",
                "tests/test_calc.py::test_skipped_feature",
            ]
        );
        for tc in &discovered {
            assert_eq!(tc.file_path, "tests/test_calc.py");
            assert!(!tc.test_id.contains('\\'));
        }

        // Execute full suite with PTS_FIXTURE_FAIL=1.
        let failing_runner = PytestRunner::default().with_env("PTS_FIXTURE_FAIL", "1");
        let fail_res = failing_runner
            .execute(&fixture, None)
            .expect("execute failing suite");
        assert_eq!(fail_res.exit_code, 1);
        assert_eq!(fail_res.outcomes.len(), 4);

        let pass_count = fail_res
            .outcomes
            .iter()
            .filter(|o| o.status == TestStatus::Pass)
            .count();
        let fail_count = fail_res
            .outcomes
            .iter()
            .filter(|o| o.status == TestStatus::Fail)
            .count();
        let skip_count = fail_res
            .outcomes
            .iter()
            .filter(|o| o.status == TestStatus::Skip)
            .count();
        assert_eq!(pass_count, 2);
        assert_eq!(fail_count, 1);
        assert_eq!(skip_count, 1);

        let failed = fail_res
            .outcomes
            .iter()
            .find(|o| o.status == TestStatus::Fail)
            .expect("one failed outcome");
        assert_eq!(failed.test_id, "tests/test_calc.py::test_divide_broken");
        assert!(
            failed
                .failure_message
                .as_deref()
                .is_some_and(|m| !m.is_empty()),
            "failed outcome must have non-empty failure_message"
        );

        // Execute a single selected test.
        let selected = vec!["tests/test_calc.py::test_add_positive".to_string()];
        let single_res = runner
            .execute(&fixture, Some(&selected))
            .expect("execute single test");
        assert_eq!(single_res.exit_code, 0);
        assert_eq!(single_res.outcomes.len(), 1);
        assert_eq!(
            single_res.outcomes[0].test_id,
            "tests/test_calc.py::test_add_positive"
        );
        assert_eq!(single_res.outcomes[0].status, TestStatus::Pass);
    }
}
