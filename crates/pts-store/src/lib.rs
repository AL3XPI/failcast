pub mod store;

pub use store::{ModelRecord, RunRecord, Store, TestOutcomeRecord, TestRecord};

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeSet;

    use pts_core::{CommitRecord, DiffFile, RunMode, SelectionReason, TestCase, TestStatus};
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

    #[test]
    fn history_snapshot_enforces_strict_as_of_cutoff() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("store.db");

        let mut store = Store::open(&db_path).expect("open store");
        store.migrate().expect("migrate");

        // Ingest commits at t = 100 and t = 200 (verify upsert_commits is idempotent).
        let commits = vec![
            CommitRecord {
                commit_sha: "c100".to_string(),
                parent_sha: None,
                author_hash: "auth1".to_string(),
                commit_ts: 100,
                files_changed: 2,
                lines_added: 10,
                lines_deleted: 0,
                files: vec![
                    DiffFile {
                        path: "src/calc.py".to_string(),
                        lines_added: 6,
                        lines_deleted: 0,
                    },
                    DiffFile {
                        path: "tests/test_calc.py".to_string(),
                        lines_added: 4,
                        lines_deleted: 0,
                    },
                ],
            },
            CommitRecord {
                commit_sha: "c200".to_string(),
                parent_sha: Some("c100".to_string()),
                author_hash: "auth1".to_string(),
                commit_ts: 200,
                files_changed: 1,
                lines_added: 5,
                lines_deleted: 1,
                files: vec![DiffFile {
                    path: "src/calc.py".to_string(),
                    lines_added: 5,
                    lines_deleted: 1,
                }],
            },
        ];
        store.upsert_commits(&commits).expect("first upsert");
        store.upsert_commits(&commits).expect("idempotent upsert");
        assert_eq!(store.commit_count().expect("commit count"), 2);

        let tc = TestCase {
            test_id: "tests/test_calc.py::test_add".to_string(),
            file_path: "tests/test_calc.py".to_string(),
        };

        // Run 1 at t = 100: PASS (duration 120ms) with diff on src/calc.py
        let run_100 = RunRecord {
            run_id: "run-100".to_string(),
            commit_sha: Some("c100".to_string()),
            diff_fingerprint: "fp-same".to_string(),
            runner: "pytest".to_string(),
            mode: RunMode::All,
            budget_ms: None,
            started_at: 100,
            finished_at: Some(101),
            total_duration_ms: Some(120),
            exit_code: Some(0),
        };
        store
            .persist_run_with_outcomes(
                &run_100,
                &[DiffFile {
                    path: "src/calc.py".to_string(),
                    lines_added: 6,
                    lines_deleted: 0,
                }],
                std::slice::from_ref(&tc),
                &[TestOutcomeRecord {
                    run_id: "run-100".to_string(),
                    test_id: tc.test_id.clone(),
                    status: TestStatus::Pass,
                    duration_ms: 120,
                    predicted_prob: None,
                    selection_reason: SelectionReason::Full,
                    sampling_propensity: 1.0,
                    failure_excerpt: None,
                }],
                100,
            )
            .expect("persist run at t=100");

        // Run 2 at t = 200: FAIL (duration 400ms) on identical diff_fingerprint + new file src/future.py
        let run_200 = RunRecord {
            run_id: "run-200".to_string(),
            commit_sha: Some("c200".to_string()),
            diff_fingerprint: "fp-same".to_string(),
            runner: "pytest".to_string(),
            mode: RunMode::Budget,
            budget_ms: Some(5000),
            started_at: 200,
            finished_at: Some(201),
            total_duration_ms: Some(400),
            exit_code: Some(1),
        };
        store
            .persist_run_with_outcomes(
                &run_200,
                &[
                    DiffFile {
                        path: "src/calc.py".to_string(),
                        lines_added: 5,
                        lines_deleted: 1,
                    },
                    DiffFile {
                        path: "src/future.py".to_string(),
                        lines_added: 20,
                        lines_deleted: 0,
                    },
                ],
                std::slice::from_ref(&tc),
                &[TestOutcomeRecord {
                    run_id: "run-200".to_string(),
                    test_id: tc.test_id.clone(),
                    status: TestStatus::Fail,
                    duration_ms: 400,
                    predicted_prob: Some(0.75),
                    selection_reason: SelectionReason::Heuristic,
                    sampling_propensity: 1.0,
                    failure_excerpt: Some("boom".to_string()),
                }],
                200,
            )
            .expect("persist run at t=200");

        // Snapshot at t = 150 MUST strictly exclude all t = 200 data.
        let snap_150 = store.load_history_snapshot(150).expect("snapshot at 150");
        assert_eq!(snap_150.as_of_ts, 150);
        assert_eq!(snap_150.total_runs, 1);
        assert_eq!(snap_150.total_failures, 0);
        assert_eq!(snap_150.median_repo_test_ms, Some(120.0));

        let th_150 = snap_150
            .test_history
            .get(&tc.test_id)
            .expect("test history at 150");
        assert_eq!(th_150.total_runs, 1);
        assert_eq!(th_150.total_failures, 0);
        assert!((th_150.ewma_fail_10 - 0.0).abs() < 1e-12);
        assert!((th_150.ewma_fail_50 - 0.0).abs() < 1e-12);
        assert_eq!(th_150.runs_since_last_fail, None);
        assert_eq!(th_150.median_duration_ms, Some(120.0));
        assert!((th_150.flakiness_rate - 0.0).abs() < 1e-12);
        assert_eq!(th_150.last_run_ts, Some(100));

        // Co-failure counts must exclude t = 200 (src/future.py absent, src/calc.py has 0 fails / 1 run).
        assert!(!snap_150
            .cofail_counts
            .contains_key(&(tc.test_id.clone(), "src/future.py".to_string())));
        let calc_cofail = snap_150
            .cofail_counts
            .get(&(tc.test_id.clone(), "src/calc.py".to_string()))
            .expect("calc cofail at 150");
        assert_eq!(calc_cofail.co_runs, 1);
        assert_eq!(calc_cofail.co_fails, 0);

        // Git churn and co-change must exclude commit c200 at t = 200.
        assert_eq!(snap_150.churn_30d("src/calc.py"), 1);
        assert!((snap_150.cochange_freq("tests/test_calc.py", "src/calc.py") - 1.0).abs() < 1e-12);

        // Strict inequality check: snapshot at t = 100 excludes t = 100 as well.
        let snap_100 = store.load_history_snapshot(100).expect("snapshot at 100");
        assert_eq!(snap_100.total_runs, 0);
        assert!(snap_100.test_history.is_empty());
        assert_eq!(snap_100.churn_30d("src/calc.py"), 0);

        // Snapshot at t = 250 includes both t = 100 and t = 200.
        let snap_250 = store.load_history_snapshot(250).expect("snapshot at 250");
        assert_eq!(snap_250.total_runs, 2);
        assert_eq!(snap_250.total_failures, 1);
        assert_eq!(snap_250.churn_30d("src/calc.py"), 2);
        assert!((snap_250.cochange_freq("tests/test_calc.py", "src/calc.py") - 0.5).abs() < 1e-12);
        let th_250 = snap_250
            .test_history
            .get(&tc.test_id)
            .expect("test history at 250");
        assert_eq!(th_250.total_runs, 2);
        assert_eq!(th_250.total_failures, 1);
        assert_eq!(th_250.runs_since_last_fail, Some(0));
        assert_eq!(th_250.median_duration_ms, Some(260.0));
        assert!((th_250.flakiness_rate - 1.0).abs() < 1e-12);
        assert_eq!(th_250.last_run_ts, Some(200));
    }
}
