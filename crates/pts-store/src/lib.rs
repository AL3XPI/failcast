pub mod store;

pub use store::{RunRecord, Store, TestOutcomeRecord, TestRecord};

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeSet;

    use pts_core::{DiffFile, RunMode, SelectionReason, TestCase, TestStatus};
    use rusqlite::params;

    use super::*;

    #[test]
    fn schema_init_is_idempotent_and_enables_wal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("store.db");

        let store = Store::open(&db_path).expect("open store");
        store.migrate().expect("first migrate");
        store.migrate().expect("second migrate");

        let journal_mode: String = store
            .connection()
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .expect("query journal_mode");
        assert_eq!(journal_mode.to_lowercase(), "wal");

        let foreign_keys: i64 = store
            .connection()
            .query_row("PRAGMA foreign_keys", [], |r| r.get(0))
            .expect("query foreign_keys");
        assert_eq!(foreign_keys, 1);

        let mut stmt = store
            .connection()
            .prepare(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            )
            .expect("prepare tables query");
        let tables: BTreeSet<String> = stmt
            .query_map([], |r| r.get(0))
            .expect("query tables")
            .map(|r| r.expect("table row"))
            .collect();

        let expected: BTreeSet<String> = [
            "schema_migrations",
            "commits",
            "commit_files",
            "tests",
            "runs",
            "run_diff_files",
            "test_outcomes",
            "models",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(tables, expected);

        let versions: Vec<i64> = {
            let mut s = store
                .connection()
                .prepare("SELECT version FROM schema_migrations")
                .expect("prepare migrations query");
            s.query_map([], |r| r.get(0))
                .expect("query migrations")
                .map(|r| r.expect("version row"))
                .collect()
        };
        assert_eq!(versions, vec![1]);
    }

    #[test]
    fn roundtrip_run_and_outcomes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("store.db");

        let store = Store::open(&db_path).expect("open store");
        store.migrate().expect("migrate");

        let run = RunRecord {
            run_id: "run-001".to_string(),
            commit_sha: Some("abc1234".to_string()),
            diff_fingerprint: "deadbeef".to_string(),
            runner: "pytest".to_string(),
            mode: RunMode::All,
            budget_ms: None,
            started_at: 1_700_000_000,
            finished_at: Some(1_700_000_002),
            total_duration_ms: Some(1250),
            exit_code: Some(1),
        };
        store.insert_run(&run).expect("insert run");

        let diff_files = vec![
            DiffFile {
                path: "src/calc.py".to_string(),
                lines_added: 12,
                lines_deleted: 3,
            },
            DiffFile {
                path: "tests/test_calc.py".to_string(),
                lines_added: 5,
                lines_deleted: 0,
            },
        ];
        store
            .insert_run_diff_files(&run.run_id, &diff_files)
            .expect("insert diff files");
        assert_eq!(
            store
                .get_run_diff_files(&run.run_id)
                .expect("get diff files"),
            diff_files
        );

        let tests = vec![
            TestCase {
                test_id: "tests/test_calc.py::test_a".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
            TestCase {
                test_id: "tests/test_calc.py::test_b".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
            TestCase {
                test_id: "tests/test_calc.py::test_c".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
        ];
        for t in &tests {
            store.upsert_test(t, 1_700_000_000).expect("upsert test");
        }

        let outcomes = vec![
            TestOutcomeRecord {
                run_id: "run-001".to_string(),
                test_id: "tests/test_calc.py::test_a".to_string(),
                status: TestStatus::Pass,
                duration_ms: 42,
                predicted_prob: Some(0.15),
                selection_reason: SelectionReason::Ranked,
                sampling_propensity: 1.0,
                failure_excerpt: None,
            },
            TestOutcomeRecord {
                run_id: "run-001".to_string(),
                test_id: "tests/test_calc.py::test_b".to_string(),
                status: TestStatus::Fail,
                duration_ms: 130,
                predicted_prob: Some(0.82),
                selection_reason: SelectionReason::Heuristic,
                sampling_propensity: 1.0,
                failure_excerpt: Some("AssertionError: boom".to_string()),
            },
            TestOutcomeRecord {
                run_id: "run-001".to_string(),
                test_id: "tests/test_calc.py::test_c".to_string(),
                status: TestStatus::Skip,
                duration_ms: 0,
                predicted_prob: None,
                selection_reason: SelectionReason::Explore,
                sampling_propensity: 0.25,
                failure_excerpt: None,
            },
        ];
        for o in &outcomes {
            store.insert_test_outcome(o).expect("insert outcome");
        }

        let by_run = store
            .get_outcomes_by_run("run-001")
            .expect("get outcomes by run");
        assert_eq!(by_run.len(), 3);
        for (actual, expected) in by_run.iter().zip(outcomes.iter()) {
            assert_eq!(actual.test_id, expected.test_id);
            assert_eq!(actual.status, expected.status);
            assert_eq!(actual.duration_ms, expected.duration_ms);
            assert_eq!(actual.selection_reason, expected.selection_reason);
            assert!(
                (actual.sampling_propensity - expected.sampling_propensity).abs() < f64::EPSILON
            );
            assert_eq!(actual.failure_excerpt, expected.failure_excerpt);
        }

        let by_test = store
            .get_outcomes_by_test("tests/test_calc.py::test_b")
            .expect("get outcomes by test");
        assert_eq!(by_test.len(), 1);
        assert_eq!(by_test[0].status, TestStatus::Fail);
        assert_eq!(by_test[0].duration_ms, 130);
        assert_eq!(by_test[0].selection_reason, SelectionReason::Heuristic);
        assert!((by_test[0].sampling_propensity - 1.0).abs() < f64::EPSILON);

        // Invalid status string violates CHECK constraint.
        let invalid_res = store.connection().execute(
            "INSERT INTO test_outcomes (run_id, test_id, status, duration_ms, selection_reason, sampling_propensity)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params!["run-001", "tests/test_calc.py::test_a", "flaky", 10, "full", 1.0],
        );
        assert!(
            invalid_res.is_err(),
            "invalid status must violate CHECK constraint"
        );
    }
}
