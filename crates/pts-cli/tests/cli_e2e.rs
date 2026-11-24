mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use assert_cmd::cargo::cargo_bin_cmd;
    use rusqlite::Connection;

    fn git_cmd(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "PTS E2E")
            .env("GIT_AUTHOR_EMAIL", "e2e@example.com")
            .env("GIT_COMMITTER_NAME", "PTS E2E")
            .env("GIT_COMMITTER_EMAIL", "e2e@example.com")
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn copy_dir_recursive(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).expect("create dst dir");
        for entry in fs::read_dir(src).expect("read src dir") {
            let entry = entry.expect("dir entry");
            let file_type = entry.file_type().expect("file type");
            let name = entry.file_name();
            if name == "__pycache__" || name == ".pytest_cache" || name == ".pts" {
                continue;
            }
            let src_path = entry.path();
            let dst_path = dst.join(name);
            if file_type.is_dir() {
                copy_dir_recursive(&src_path, &dst_path);
            } else {
                fs::copy(&src_path, &dst_path).expect("copy fixture file");
            }
        }
    }

    fn fixture_repo_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/mini_py_repo")
    }

    #[test]
    fn cli_run_all_persists_even_on_test_failure() {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        // 1. Run `pts init`.
        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        // 2. Run `pts run --all` with PTS_FIXTURE_FAIL=1; must exit with code 1.
        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .env("PTS_FIXTURE_FAIL", "1")
            .args(["run", "--all"])
            .assert()
            .code(1);

        // 3. Open .pts/store.db directly and verify persisted run and outcomes.
        let db_path = tmp.path().join(".pts").join("store.db");
        assert!(db_path.is_file(), "store.db must exist");

        let conn = Connection::open(&db_path).expect("open store.db");

        let runs: Vec<(String, Option<i32>, String)> = {
            let mut stmt = conn
                .prepare("SELECT run_id, exit_code, mode FROM runs")
                .expect("prepare runs query");
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .expect("query runs")
                .map(|r| r.expect("run row"))
                .collect()
        };
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1, Some(1));
        assert_eq!(runs[0].2, "all");

        let run_id = &runs[0].0;
        let outcomes: Vec<(String, String, String)> = {
            let mut stmt = conn
                .prepare(
                    "SELECT test_id, status, selection_reason FROM test_outcomes WHERE run_id = ?1 ORDER BY test_id ASC",
                )
                .expect("prepare outcomes query");
            stmt.query_map([run_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .expect("query outcomes")
                .map(|r| r.expect("outcome row"))
                .collect()
        };
        assert_eq!(outcomes.len(), 4);
        assert_eq!(
            outcomes,
            vec![
                (
                    "tests/test_calc.py::test_add_negative".to_string(),
                    "pass".to_string(),
                    "full".to_string(),
                ),
                (
                    "tests/test_calc.py::test_add_positive".to_string(),
                    "pass".to_string(),
                    "full".to_string(),
                ),
                (
                    "tests/test_calc.py::test_divide_broken".to_string(),
                    "fail".to_string(),
                    "full".to_string(),
                ),
                (
                    "tests/test_calc.py::test_skipped_feature".to_string(),
                    "skip".to_string(),
                    "full".to_string(),
                ),
            ]
        );
    }

    #[test]
    fn cli_run_budget_and_dry_run_e2e() {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        // Initialize a git repository with an initial commit.
        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        // Make a working-tree edit to src/calc.py (keeps existing functions passing).
        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read src/calc.py");
        calc_src.push_str("\n\ndef multiply(a: int, b: int) -> int:\n    return a * b\n");
        fs::write(&calc_path, calc_src).expect("modify src/calc.py");

        // 1. Run `pts init` and verify git history was mined into SQLite.
        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        let db_path = tmp.path().join(".pts").join("store.db");
        assert!(db_path.is_file(), "store.db must exist");

        let conn = Connection::open(&db_path).expect("open store.db");
        let commit_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM commits", [], |r| r.get(0))
            .expect("count commits");
        assert_eq!(commit_count, 1, "pts init must mine initial git commit");

        // 2. Run `pts run --budget 5 --dry-run --json`: exits 0, writes 0 runs to DB.
        let dry_output = cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .args(["run", "--budget", "5", "--dry-run", "--json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();

        let dry_json: serde_json::Value =
            serde_json::from_slice(&dry_output).expect("parse dry-run JSON");
        assert_eq!(dry_json["dry_run"], true);
        assert_eq!(dry_json["mode"], "budget");
        assert_eq!(dry_json["model_mode"], "heuristic");
        assert_eq!(dry_json["diff_files_count"], 1);
        assert_eq!(dry_json["total_discovered_tests"], 4);
        assert!(dry_json["selected_count"].as_u64().unwrap_or(0) >= 1);

        let runs_after_dry: i64 = conn
            .query_row("SELECT COUNT(*) FROM runs", [], |r| r.get(0))
            .expect("count runs after dry run");
        assert_eq!(runs_after_dry, 0, "dry-run must write 0 runs to DB");

        // 3. Run `pts run --budget 5`: executes selected tests and persists metadata.
        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .args(["run", "--budget", "5"])
            .assert()
            .success();

        struct RunRow {
            run_id: String,
            mode: String,
            budget_ms: Option<i64>,
            exit_code: Option<i32>,
            commit_sha: Option<String>,
        }

        let runs: Vec<RunRow> = {
            let mut stmt = conn
                .prepare("SELECT run_id, mode, budget_ms, exit_code, commit_sha FROM runs")
                .expect("prepare runs query");
            stmt.query_map([], |r| {
                Ok(RunRow {
                    run_id: r.get(0)?,
                    mode: r.get(1)?,
                    budget_ms: r.get(2)?,
                    exit_code: r.get(3)?,
                    commit_sha: r.get(4)?,
                })
            })
            .expect("query runs")
            .map(|r| r.expect("run row"))
            .collect()
        };
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].mode, "budget");
        assert_eq!(runs[0].budget_ms, Some(5000));
        assert_eq!(runs[0].exit_code, Some(0));
        assert!(runs[0].commit_sha.is_some(), "commit_sha must be recorded");

        let run_id = &runs[0].run_id;
        let diff_files: Vec<(String, i64, i64)> = {
            let mut stmt = conn
                .prepare(
                    "SELECT file_path, lines_added, lines_deleted FROM run_diff_files WHERE run_id = ?1",
                )
                .expect("prepare diff_files query");
            stmt.query_map([run_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .expect("query diff_files")
                .map(|r| r.expect("diff_file row"))
                .collect()
        };
        assert_eq!(diff_files.len(), 1);
        assert_eq!(diff_files[0].0, "src/calc.py");
        assert!(diff_files[0].1 >= 3);
        assert_eq!(diff_files[0].2, 0);

        let outcomes: Vec<(String, Option<f64>, String, f64)> = {
            let mut stmt = conn
                .prepare(
                    "SELECT test_id, predicted_prob, selection_reason, sampling_propensity FROM test_outcomes WHERE run_id = ?1 ORDER BY test_id ASC",
                )
                .expect("prepare outcomes query");
            stmt.query_map([run_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
                .expect("query outcomes")
                .map(|r| r.expect("outcome row"))
                .collect()
        };
        assert!(!outcomes.is_empty(), "selected tests must be persisted");
        for (test_id, prob, reason, propensity) in &outcomes {
            let p = prob.unwrap_or_else(|| panic!("missing predicted_prob for {test_id}"));
            assert!(p > 0.0 && p < 1.0, "prob {p} out of (0,1) for {test_id}");
            assert!(
                reason == "heuristic" || reason == "explore",
                "unexpected selection_reason '{reason}' for {test_id}"
            );
            assert!(
                *propensity > 0.0 && *propensity <= 1.0,
                "invalid sampling_propensity {propensity} for {test_id}"
            );
        }
    }

    #[test]
    fn cli_train_run_and_eval_e2e() {
        use pts_core::{DiffFile, DiffSummary, RunMode, SelectionReason, TestCase, TestStatus};
        use pts_store::{RunRecord, Store, TestOutcomeRecord};

        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        // 1. Initialize store.
        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        // 2. Populate DB with 105 chronological synthetic runs (>= default min_runs_for_ml=100
        //    and >= min_failures_for_ml=25) so `pts run` automatically activates GBDT mode.
        let db_path = tmp.path().join(".pts").join("store.db");
        let mut store = Store::open(&db_path).expect("open store");
        store.migrate().expect("migrate");

        let tests = vec![
            TestCase {
                test_id: "tests/test_calc.py::test_add_positive".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
            TestCase {
                test_id: "tests/test_calc.py::test_add_negative".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
            TestCase {
                test_id: "tests/test_calc.py::test_divide_broken".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
            TestCase {
                test_id: "tests/test_calc.py::test_skipped_feature".to_string(),
                file_path: "tests/test_calc.py".to_string(),
            },
        ];

        for r in 0..105usize {
            let started_at = 1_700_000_000 + (r as i64) * 600;
            let touches_calc = r % 2 == 0;
            let is_bug = touches_calc && (r % 4 == 0 || r % 6 == 0);
            let diff_files = if touches_calc {
                vec![DiffFile {
                    path: "src/calc.py".to_string(),
                    lines_added: if is_bug { 18 } else { 3 },
                    lines_deleted: 2,
                }]
            } else {
                vec![DiffFile {
                    path: "README.md".to_string(),
                    lines_added: 2,
                    lines_deleted: 1,
                }]
            };
            let diff_fp = DiffSummary::compute_fingerprint(&diff_files);
            let run_id = format!("hist-run-{r:04}");
            let fail_divide = is_bug;

            let run_rec = RunRecord {
                run_id: run_id.clone(),
                commit_sha: Some(format!("sha-{r:04}")),
                diff_fingerprint: diff_fp,
                runner: "pytest".to_string(),
                mode: RunMode::All,
                budget_ms: None,
                started_at,
                finished_at: Some(started_at + 1),
                total_duration_ms: Some(450),
                exit_code: Some(if fail_divide { 1 } else { 0 }),
            };

            let outcomes = vec![
                TestOutcomeRecord {
                    run_id: run_id.clone(),
                    test_id: tests[0].test_id.clone(),
                    status: TestStatus::Pass,
                    duration_ms: 80,
                    predicted_prob: None,
                    selection_reason: SelectionReason::Full,
                    sampling_propensity: 1.0,
                    failure_excerpt: None,
                },
                TestOutcomeRecord {
                    run_id: run_id.clone(),
                    test_id: tests[1].test_id.clone(),
                    status: TestStatus::Pass,
                    duration_ms: 90,
                    predicted_prob: None,
                    selection_reason: SelectionReason::Full,
                    sampling_propensity: 1.0,
                    failure_excerpt: None,
                },
                TestOutcomeRecord {
                    run_id: run_id.clone(),
                    test_id: tests[2].test_id.clone(),
                    status: if fail_divide {
                        TestStatus::Fail
                    } else {
                        TestStatus::Pass
                    },
                    duration_ms: 140,
                    predicted_prob: None,
                    selection_reason: SelectionReason::Full,
                    sampling_propensity: 1.0,
                    failure_excerpt: if fail_divide {
                        Some("AssertionError: divide broken".to_string())
                    } else {
                        None
                    },
                },
                TestOutcomeRecord {
                    run_id: run_id.clone(),
                    test_id: tests[3].test_id.clone(),
                    status: TestStatus::Skip,
                    duration_ms: 1,
                    predicted_prob: None,
                    selection_reason: SelectionReason::Full,
                    sampling_propensity: 1.0,
                    failure_excerpt: None,
                },
            ];

            store
                .persist_run_with_outcomes(&run_rec, &diff_files, &tests, &outcomes, started_at)
                .expect("persist synthetic run");
        }
        drop(store);

        // 3. Execute `pts train` and verify `.pts/model.json` and `models` SQLite row.
        let train_out = cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .arg("train")
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let train_stdout = String::from_utf8_lossy(&train_out);
        assert!(train_stdout.contains("Platt:"), "train summary must include Platt params");
        assert!(train_stdout.contains("Brier:"), "train summary must include Brier score");

        let model_path = tmp.path().join(".pts").join("model.json");
        assert!(model_path.is_file(), ".pts/model.json must be written");
        let model_raw = fs::read_to_string(&model_path).expect("read model.json");
        let loaded = pts_ml::TrainedModel::from_json(&model_raw).expect("valid ModelArtifact JSON");
        assert!(loaded.artifact.train_rows > 0);
        assert!(loaded.artifact.pos_rows > 0);
        assert!(loaded.artifact.platt_a >= 0.0);

        let conn = Connection::open(&db_path).expect("open store.db");
        let model_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM models", [], |r| r.get(0))
            .expect("count models");
        assert_eq!(model_rows, 1, "models table must contain 1 trained model row");

        // 4. Modify src/calc.py and run `pts run --budget 5 --dry-run --json` -> assert model_mode == "gbdt".
        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read src/calc.py");
        calc_src.push_str("\n\ndef sub(a: int, b: int) -> int:\n    return a - b\n");
        fs::write(&calc_path, calc_src).expect("modify src/calc.py");

        let dry_out = cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .args(["run", "--budget", "5", "--dry-run", "--json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let dry_json: serde_json::Value =
            serde_json::from_slice(&dry_out).expect("parse dry-run JSON");
        assert_eq!(dry_json["model_mode"], "gbdt");
        assert!(dry_json["selected_count"].as_u64().unwrap_or(0) >= 1);

        // 5. Run `pts eval --output-json report.json` and verify valid JSON and Markdown output.
        let eval_out = cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .args(["eval", "--output-json", "report.json", "--retrain-every", "25"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let eval_md = String::from_utf8_lossy(&eval_out);
        assert!(eval_md.contains("| Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |"));
        assert!(eval_md.contains("PTS (GBDT+Platt)"));
        assert!(eval_md.contains("Random"));
        assert!(eval_md.contains("StaticAst"));
        assert!(eval_md.contains("RecentFailureRate"));

        let report_path = tmp.path().join("report.json");
        assert!(report_path.is_file(), "report.json must be written");
        let report_raw = fs::read_to_string(&report_path).expect("read report.json");
        let report_json: serde_json::Value =
            serde_json::from_str(&report_raw).expect("valid report.json");
        assert_eq!(report_json["zero_leakage_verified"], true);
        assert!(report_json["budgets"].as_array().is_some_and(|b| b.len() == 3));
    }

    #[test]
    fn cli_backfill_uses_isolated_worktree_and_skips_broken_commits() {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "commit 1: clean"]);

        // Commit 2: introduce a syntax error in tests/test_calc.py (should be gracefully skipped).
        let test_path = tmp.path().join("tests/test_calc.py");
        let orig_test_src = fs::read_to_string(&test_path).expect("read test_calc.py");
        fs::write(&test_path, "def broken(:\n    pass\n").expect("write broken test");
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "commit 2: broken syntax"]);

        // Commit 3: restore valid tests and add a helper function.
        fs::write(&test_path, &orig_test_src).expect("restore test_calc.py");
        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read calc.py");
        calc_src.push_str("\ndef square(x: int) -> int:\n    return x * x\n");
        fs::write(&calc_path, &calc_src).expect("update calc.py");
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "commit 3: fixed + square"]);

        // Leave an uncommitted local edit in user's working tree to ensure backfill NEVER touches it.
        let marker_edit = format!("{calc_src}\n# uncommitted user edit\n");
        fs::write(&calc_path, &marker_edit).expect("write uncommitted edit");

        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        cargo_bin_cmd!("pts")
            .current_dir(tmp.path())
            .args(["backfill", "--commits", "3"])
            .assert()
            .success();

        // User's working tree must remain completely untouched.
        assert_eq!(
            fs::read_to_string(&calc_path).expect("read calc.py after backfill"),
            marker_edit
        );

        // Exactly 2 valid commits persisted with mode = 'backfill', 1 broken commit skipped.
        let db_path = tmp.path().join(".pts").join("store.db");
        let conn = Connection::open(&db_path).expect("open store.db");
        let backfill_runs: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE mode = 'backfill'",
                [],
                |r| r.get(0),
            )
            .expect("count backfill runs");
        assert_eq!(backfill_runs, 2, "expected 2 valid commits backfilled and 1 broken skipped");

        // No lingering git worktrees besides the main repo.
        let wt_out = Command::new("git")
            .args(["worktree", "list", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .expect("git worktree list");
        let wt_count = String::from_utf8_lossy(&wt_out.stdout)
            .lines()
            .filter(|l| l.starts_with("worktree "))
            .count();
        assert_eq!(wt_count, 1, "temporary backfill worktree must be cleaned up");
    }
}
