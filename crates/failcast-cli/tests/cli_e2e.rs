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
            .env("GIT_AUTHOR_NAME", "Failcast E2E")
            .env("GIT_AUTHOR_EMAIL", "e2e@example.com")
            .env("GIT_COMMITTER_NAME", "Failcast E2E")
            .env("GIT_COMMITTER_EMAIL", "e2e@example.com")
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
        if args.first() == Some(&"init") {
            let cfg_out = Command::new("git")
                .args(["config", "core.autocrlf", "false"])
                .current_dir(dir)
                .output()
                .expect("configure git core.autocrlf");
            assert!(cfg_out.status.success(), "git config core.autocrlf failed");
        }
    }

    fn copy_dir_recursive(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).expect("create dst dir");
        for entry in fs::read_dir(src).expect("read src dir") {
            let entry = entry.expect("dir entry");
            let file_type = entry.file_type().expect("file type");
            let name = entry.file_name();
            if name == "__pycache__" || name == ".pytest_cache" || name == ".failcast" {
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

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .env("FAILCAST_FIXTURE_FAIL", "1")
            .args(["run", "--all"])
            .assert()
            .code(1);

        let db_path = tmp.path().join(".failcast").join("store.db");
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

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read src/calc.py");
        calc_src.push_str("\n\ndef multiply(a: int, b: int) -> int:\n    return a * b\n");
        fs::write(&calc_path, calc_src).expect("modify src/calc.py");

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        let db_path = tmp.path().join(".failcast").join("store.db");
        assert!(db_path.is_file(), "store.db must exist");

        let conn = Connection::open(&db_path).expect("open store.db");
        let commit_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM commits", [], |r| r.get(0))
            .expect("count commits");
        assert_eq!(
            commit_count, 1,
            "failcast init must mine initial git commit"
        );

        let dry_output = cargo_bin_cmd!("failcast")
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

        cargo_bin_cmd!("failcast")
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
            stmt.query_map([run_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })
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
        use failcast_core::{
            DiffFile, DiffSummary, RunMode, SelectionReason, TestCase, TestStatus,
        };
        use failcast_store::{RunRecord, Store, TestOutcomeRecord};

        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        // Populate 105 runs to cross default min_runs_for_ml=100 and min_failures_for_ml=25.
        let db_path = tmp.path().join(".failcast").join("store.db");
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

        let train_out = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("train")
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let train_stdout = String::from_utf8_lossy(&train_out);
        assert!(
            train_stdout.contains("Platt:"),
            "train summary must include Platt params"
        );
        assert!(
            train_stdout.contains("Brier:"),
            "train summary must include Brier score"
        );

        let model_path = tmp.path().join(".failcast").join("model.json");
        assert!(model_path.is_file(), ".failcast/model.json must be written");
        let model_raw = fs::read_to_string(&model_path).expect("read model.json");
        let loaded =
            failcast_ml::TrainedModel::from_json(&model_raw).expect("valid ModelArtifact JSON");
        assert!(loaded.artifact.train_rows > 0);
        assert!(loaded.artifact.pos_rows > 0);
        assert!(loaded.artifact.platt_a >= 0.0);

        let conn = Connection::open(&db_path).expect("open store.db");
        let model_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM models", [], |r| r.get(0))
            .expect("count models");
        assert_eq!(
            model_rows, 1,
            "models table must contain 1 trained model row"
        );

        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read src/calc.py");
        calc_src.push_str("\n\ndef sub(a: int, b: int) -> int:\n    return a - b\n");
        fs::write(&calc_path, calc_src).expect("modify src/calc.py");

        let dry_out = cargo_bin_cmd!("failcast")
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

        let eval_out = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args([
                "eval",
                "--output-json",
                "report.json",
                "--retrain-every",
                "25",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let eval_md = String::from_utf8_lossy(&eval_out);
        assert!(eval_md.contains(
            "| Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |"
        ));
        assert!(eval_md.contains("Failcast (GBDT+Platt)"));
        assert!(eval_md.contains("Random"));
        assert!(eval_md.contains("StaticAst"));
        assert!(eval_md.contains("RecentFailureRate"));

        let report_path = tmp.path().join("report.json");
        assert!(report_path.is_file(), "report.json must be written");
        let report_raw = fs::read_to_string(&report_path).expect("read report.json");
        let report_json: serde_json::Value =
            serde_json::from_str(&report_raw).expect("valid report.json");
        assert_eq!(report_json["zero_leakage_verified"], true);
        assert!(
            report_json["budgets"]
                .as_array()
                .is_some_and(|b| b.len() == 3)
        );
    }

    #[test]
    fn cli_backfill_uses_isolated_worktree_and_skips_broken_commits() {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "commit 1: clean"]);

        let test_path = tmp.path().join("tests/test_calc.py");
        let orig_test_src = fs::read_to_string(&test_path).expect("read test_calc.py");
        fs::write(&test_path, "def broken(:\n    pass\n").expect("write broken test");
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "commit 2: broken syntax"]);

        fs::write(&test_path, &orig_test_src).expect("restore test_calc.py");
        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read calc.py");
        calc_src.push_str("\ndef square(x: int) -> int:\n    return x * x\n");
        fs::write(&calc_path, &calc_src).expect("update calc.py");
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "commit 3: fixed + square"]);

        let marker_edit = format!("{calc_src}\n# uncommitted user edit\n");
        fs::write(&calc_path, &marker_edit).expect("write uncommitted edit");

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["backfill", "--commits", "3"])
            .assert()
            .success();

        assert_eq!(
            fs::read_to_string(&calc_path).expect("read calc.py after backfill"),
            marker_edit
        );

        let db_path = tmp.path().join(".failcast").join("store.db");
        let conn = Connection::open(&db_path).expect("open store.db");
        let backfill_runs: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE mode = 'backfill'",
                [],
                |r| r.get(0),
            )
            .expect("count backfill runs");
        assert_eq!(
            backfill_runs, 2,
            "expected 2 valid commits backfilled and 1 broken skipped"
        );

        let wt_out = Command::new("git")
            .args(["worktree", "list", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .expect("git worktree list");
        let wt_count = String::from_utf8_lossy(&wt_out.stdout)
            .lines()
            .filter(|l| l.starts_with("worktree "))
            .count();
        assert_eq!(
            wt_count, 1,
            "temporary backfill worktree must be cleaned up"
        );
    }

    #[test]
    fn cli_explain_outputs_features_and_rationale() {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read src/calc.py");
        calc_src.push_str("\n\ndef power(a: int, b: int) -> int:\n    return a ** b\n");
        fs::write(&calc_path, calc_src).expect("modify src/calc.py");

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        let json_out = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["explain", "tests/test_calc.py::test_add_positive", "--json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();

        let report: serde_json::Value =
            serde_json::from_slice(&json_out).expect("parse failcast explain --json output");
        assert_eq!(report["test_id"], "tests/test_calc.py::test_add_positive");
        assert_eq!(report["model_mode"], "heuristic");

        let prob = report["predicted_fail_prob"]
            .as_f64()
            .expect("predicted_fail_prob must be numeric");
        assert!(
            prob > 0.0 && prob < 1.0,
            "predicted_fail_prob out of (0,1): {prob}"
        );

        let yield_score = report["yield_score"]
            .as_f64()
            .expect("yield_score must be numeric");
        assert!(yield_score > 0.0, "yield_score must be positive");
        assert!(
            report["est_duration_ms"].as_u64().unwrap_or(0) >= 50,
            "est_duration_ms must respect 50ms floor"
        );

        let features = report["features"]
            .as_object()
            .expect("features must be an object");
        assert_eq!(features.len(), 16, "must include all 16 features");
        assert_eq!(
            features["ast_min_hop"].as_f64(),
            Some(1.0),
            "expected ast_min_hop == 1.0 for direct import of src/calc.py"
        );
        assert_eq!(features["direct_test_modified"].as_f64(), Some(0.0));
        assert_eq!(features["diff_files_count"].as_f64(), Some(1.0));

        let signals = report["top_signals"]
            .as_array()
            .expect("top_signals must be an array");
        assert!(
            signals.iter().any(|s| s["feature"] == "ast_min_hop"),
            "top_signals must highlight ast_min_hop"
        );

        let text_out = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["explain", "tests/test_calc.py::test_add_positive"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let text = String::from_utf8_lossy(&text_out);
        assert!(text.contains("Top Contributing Signals:"));
        assert!(text.contains("16-Feature Breakdown:"));
        assert!(text.contains("ast_min_hop"));

        let err_out = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["explain", "tests/test_calc.py::test_does_not_exist"])
            .assert()
            .failure()
            .get_output()
            .stderr
            .clone();
        let err_text = String::from_utf8_lossy(&err_out);
        assert!(
            err_text.contains("not found"),
            "expected clear not-found error on stderr, got: {err_text}"
        );
    }

    #[test]
    fn cli_mcp_stdio_jsonrpc_roundtrip_and_stdout_purity() {
        use std::io::{BufRead, BufReader, Write};
        use std::process::Stdio;

        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        let calc_path = tmp.path().join("src/calc.py");
        let mut calc_src = fs::read_to_string(&calc_path).expect("read src/calc.py");
        calc_src.push_str("\n\ndef negate(a: int) -> int:\n    return -a\n");
        fs::write(&calc_path, calc_src).expect("modify src/calc.py");

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        let mut child = Command::new(env!("CARGO_BIN_EXE_failcast"))
            .arg("mcp")
            .current_dir(tmp.path())
            .env("FAILCAST_FIXTURE_FAIL", "1")
            .env("RUST_LOG", "info")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn `failcast mcp`");

        let mut stdin = child.stdin.take().expect("child stdin");
        let stdout = child.stdout.take().expect("child stdout");
        let mut reader = BufReader::new(stdout);

        let mut all_stdout_lines: Vec<String> = Vec::new();

        let send_msg = |writer: &mut std::process::ChildStdin, payload: &serde_json::Value| {
            let line = serde_json::to_string(payload).expect("serialize JSON-RPC message");
            writeln!(writer, "{line}").expect("write JSON-RPC line to stdin");
            writer.flush().expect("flush stdin");
        };

        let read_jsonrpc = |rdr: &mut BufReader<std::process::ChildStdout>,
                            log: &mut Vec<String>|
         -> serde_json::Value {
            loop {
                let mut line = String::new();
                let n = rdr.read_line(&mut line).expect("read line from stdout");
                assert!(n > 0, "unexpected EOF from `failcast mcp` stdout");
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                log.push(trimmed.to_string());
                let val: serde_json::Value = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                    panic!("non-JSON-RPC line leaked on stdout: `{trimmed}` ({e})")
                });
                assert_eq!(
                    val["jsonrpc"], "2.0",
                    "stdout line is not a JSON-RPC 2.0 frame: `{trimmed}`"
                );
                return val;
            }
        };

        send_msg(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "failcast-e2e-test", "version": "0.1.0" }
                }
            }),
        );
        let init_resp = read_jsonrpc(&mut reader, &mut all_stdout_lines);
        assert_eq!(init_resp["id"], 1);
        assert!(
            init_resp.get("error").is_none(),
            "initialize failed: {init_resp}"
        );
        let caps = init_resp["result"]["capabilities"]
            .as_object()
            .expect("server capabilities must be an object");
        assert!(
            caps.get("tools").is_some_and(|v| v.is_object()),
            "server must advertise tools capability: {init_resp}"
        );
        assert!(
            caps.get("resources").is_none(),
            "server must not advertise resources capability: {init_resp}"
        );
        assert!(
            caps.get("prompts").is_none(),
            "server must not advertise prompts capability: {init_resp}"
        );
        assert_eq!(init_resp["result"]["serverInfo"]["name"], "failcast");
        assert_eq!(
            init_resp["result"]["serverInfo"]["version"],
            env!("CARGO_PKG_VERSION")
        );

        send_msg(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
        );

        send_msg(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }),
        );
        let list_resp = read_jsonrpc(&mut reader, &mut all_stdout_lines);
        assert_eq!(list_resp["id"], 2);
        let tools = list_resp["result"]["tools"]
            .as_array()
            .expect("tools/list must return tools array");
        assert_eq!(tools.len(), 3, "expected exactly 3 MCP tools");

        let expected_tools = [
            ("select_tests", true, false, true, false),
            ("run_fast_tests", false, false, false, false),
            ("explain_selection", true, false, true, false),
        ];
        for (expected_name, read_only, destructive, idempotent, open_world) in expected_tools {
            let tool = tools
                .iter()
                .find(|t| t["name"] == expected_name)
                .unwrap_or_else(|| panic!("missing tool `{expected_name}` in tools/list"));
            assert!(
                tool["inputSchema"].is_object(),
                "tool `{expected_name}` must expose valid inputSchema"
            );
            assert!(
                tool["outputSchema"].is_object(),
                "tool `{expected_name}` must expose valid outputSchema"
            );
            let annotations = tool["annotations"].as_object().unwrap_or_else(|| {
                panic!("tool `{expected_name}` must expose annotations: {tool}")
            });
            assert_eq!(
                annotations.get("readOnlyHint").and_then(|v| v.as_bool()),
                Some(read_only),
                "unexpected readOnlyHint for `{expected_name}`: {tool}"
            );
            assert_eq!(
                annotations.get("destructiveHint").and_then(|v| v.as_bool()),
                Some(destructive),
                "unexpected destructiveHint for `{expected_name}`: {tool}"
            );
            assert_eq!(
                annotations.get("idempotentHint").and_then(|v| v.as_bool()),
                Some(idempotent),
                "unexpected idempotentHint for `{expected_name}`: {tool}"
            );
            assert_eq!(
                annotations.get("openWorldHint").and_then(|v| v.as_bool()),
                Some(open_world),
                "unexpected openWorldHint for `{expected_name}`: {tool}"
            );
        }

        let extract_tool_json = |resp: &serde_json::Value| -> serde_json::Value {
            assert!(resp.get("error").is_none(), "JSON-RPC error: {resp}");
            let result = &resp["result"];
            assert_ne!(
                result["isError"],
                serde_json::Value::Bool(true),
                "tool call returned isError=true: {resp}"
            );
            if let Some(structured) = result.get("structuredContent")
                && structured.is_object()
            {
                return structured.clone();
            }
            let text = result["content"][0]["text"]
                .as_str()
                .expect("tool result must contain JSON text content");
            serde_json::from_str(text).expect("parse tool text content as JSON")
        };

        send_msg(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "select_tests",
                    "arguments": {
                        "budget_seconds": 15,
                        "top_k": null,
                        "target_recall": null
                    }
                }
            }),
        );
        let select_resp = read_jsonrpc(&mut reader, &mut all_stdout_lines);
        assert_eq!(select_resp["id"], 3);
        let select_json = extract_tool_json(&select_resp);
        assert_eq!(select_json["model_mode"], "heuristic");
        assert_eq!(select_json["diff_files_count"], 1);
        assert_eq!(select_json["total_discovered_tests"], 4);
        assert!(select_json["selected_count"].as_u64().unwrap_or(0) >= 1);

        send_msg(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "explain_selection",
                    "arguments": {
                        "test_id": "tests/test_calc.py::test_add_positive"
                    }
                }
            }),
        );
        let explain_resp = read_jsonrpc(&mut reader, &mut all_stdout_lines);
        assert_eq!(explain_resp["id"], 4);
        let explain_json = extract_tool_json(&explain_resp);
        assert_eq!(
            explain_json["test_id"],
            "tests/test_calc.py::test_add_positive"
        );
        assert_eq!(explain_json["features"]["ast_min_hop"].as_f64(), Some(1.0));
        assert!(explain_json["yield_score"].as_f64().unwrap_or(0.0) > 0.0);

        send_msg(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "run_fast_tests",
                    "arguments": {
                        "budget_seconds": 15,
                        "top_k": null,
                        "run_all": false
                    }
                }
            }),
        );
        let run_resp = read_jsonrpc(&mut reader, &mut all_stdout_lines);
        assert_eq!(run_resp["id"], 5);
        let run_json = extract_tool_json(&run_resp);
        let run_id = run_json["run_id"]
            .as_str()
            .expect("run_fast_tests must return run_id")
            .to_string();
        assert!(!run_id.is_empty());
        assert_eq!(run_json["failed"], 1, "expected 1 failing test: {run_json}");

        let failures = run_json["failures"]
            .as_array()
            .expect("failures must be an array");
        let divide_fail = failures
            .iter()
            .find(|f| f["test_id"] == "tests/test_calc.py::test_divide_broken")
            .unwrap_or_else(|| panic!("expected test_divide_broken in failures: {run_json}"));
        let fail_msg = divide_fail["failure_message"]
            .as_str()
            .expect("failing test must include failure_message");
        assert!(!fail_msg.is_empty());
        assert!(
            fail_msg.chars().count() <= 1500,
            "failure_message must be truncated to <= 1500 chars"
        );

        drop(stdin);
        let mut trailing = String::new();
        while reader.read_line(&mut trailing).expect("drain stdout") > 0 {
            let trimmed = trailing.trim();
            if !trimmed.is_empty() {
                let val: serde_json::Value = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                    panic!("non-JSON-RPC trailing line leaked on stdout: `{trimmed}` ({e})")
                });
                assert_eq!(val["jsonrpc"], "2.0");
                all_stdout_lines.push(trimmed.to_string());
            }
            trailing.clear();
        }

        let status = child.wait().expect("wait for `failcast mcp`");
        assert!(
            status.success(),
            "`failcast mcp` should exit cleanly on stdin EOF (got {status})"
        );
        assert_eq!(
            all_stdout_lines.len(),
            5,
            "expected exactly 5 JSON-RPC responses on stdout"
        );

        let db_path = tmp.path().join(".failcast").join("store.db");
        let conn = Connection::open(&db_path).expect("open store.db");
        let persisted_exit: i32 = conn
            .query_row(
                "SELECT exit_code FROM runs WHERE run_id = ?1",
                [&run_id],
                |r| r.get(0),
            )
            .expect("query persisted MCP run");
        assert_eq!(persisted_exit, 1);

        let failed_outcomes: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM test_outcomes WHERE run_id = ?1 AND status = 'fail'",
                [&run_id],
                |r| r.get(0),
            )
            .expect("count failed outcomes");
        assert_eq!(failed_outcomes, 1);
    }

    #[test]
    fn cli_import_runs_jsonl_and_eval_e2e() {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial fixture commit"]);

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        let mut jsonl = String::new();
        for r in 0..40usize {
            let started_at = 1_710_000_000 + (r as i64) * 900;
            let touches_calc = r % 2 == 0;
            let is_bug = touches_calc && (r % 4 == 0 || r % 6 == 0);

            let diff_files = if touches_calc {
                serde_json::json!([
                    { "path": "src/calc.py", "lines_added": if is_bug { 24 } else { 4 }, "lines_deleted": 2 }
                ])
            } else {
                serde_json::json!([
                    { "path": "docs/notes.md", "lines_added": 5, "lines_deleted": 1 }
                ])
            };

            let line = serde_json::json!({
                "commit_sha": format!("ci-sha-{r:04}"),
                "started_at": started_at,
                "diff_files": diff_files,
                "outcomes": [
                    {
                        "test_id": "tests/test_calc.py::test_add_positive",
                        "file_path": "tests/test_calc.py",
                        "status": "pass",
                        "duration_ms": 75
                    },
                    {
                        "test_id": "tests/test_calc.py::test_add_negative",
                        "file_path": "tests/test_calc.py",
                        "status": "pass",
                        "duration_ms": 85
                    },
                    {
                        "test_id": "tests/test_calc.py::test_divide_broken",
                        "file_path": "tests/test_calc.py",
                        "status": if is_bug { "fail" } else { "pass" },
                        "duration_ms": 130
                    },
                    {
                        "test_id": "tests/test_calc.py::test_skipped_feature",
                        "file_path": "tests/test_calc.py",
                        "status": "skip",
                        "duration_ms": 2
                    }
                ]
            });
            jsonl.push_str(&serde_json::to_string(&line).expect("serialize JSONL line"));
            jsonl.push('\n');
        }

        let jsonl_path = tmp.path().join("ci_history.jsonl");
        fs::write(&jsonl_path, &jsonl).expect("write ci_history.jsonl");

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["import-runs", "ci_history.jsonl"])
            .assert()
            .success();

        let db_path = tmp.path().join(".failcast").join("store.db");
        let conn = Connection::open(&db_path).expect("open store.db");

        let backfill_runs: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE mode = 'backfill'",
                [],
                |r| r.get(0),
            )
            .expect("count imported backfill runs");
        assert_eq!(backfill_runs, 40);

        let total_outcomes: i64 = conn
            .query_row("SELECT COUNT(*) FROM test_outcomes", [], |r| r.get(0))
            .expect("count imported outcomes");
        assert_eq!(total_outcomes, 160);

        let total_diff_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM run_diff_files", [], |r| r.get(0))
            .expect("count imported diff files");
        assert_eq!(total_diff_rows, 40);

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("train")
            .assert()
            .success();
        assert!(
            tmp.path().join(".failcast").join("model.json").is_file(),
            "model.json must be generated from imported runs"
        );

        let eval_out = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args([
                "eval",
                "--output-json",
                "report.json",
                "--min-train-runs",
                "15",
                "--retrain-every",
                "10",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let eval_md = String::from_utf8_lossy(&eval_out);
        assert!(eval_md.contains("Failcast (GBDT+Platt)"));
        assert!(eval_md.contains("Random"));
        assert!(eval_md.contains("StaticAst"));
        assert!(eval_md.contains("RecentFailureRate"));

        let report_path = tmp.path().join("report.json");
        assert!(report_path.is_file(), "report.json must be written");
        let report_raw = fs::read_to_string(&report_path).expect("read report.json");
        let report_json: serde_json::Value =
            serde_json::from_str(&report_raw).expect("valid report.json");
        assert_eq!(report_json["zero_leakage_verified"], true);
        assert_eq!(report_json["evaluated_runs"], 25);
        assert!(
            report_json["budgets"]
                .as_array()
                .is_some_and(|b| b.len() == 3)
        );
    }

    #[test]
    fn cli_discovery_cache_hits_and_invalidates_on_test_change() {
        #[cfg(unix)]
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&fixture_repo_path(), tmp.path());

        git_cmd(tmp.path(), &["init"]);
        git_cmd(tmp.path(), &["add", "."]);
        git_cmd(tmp.path(), &["commit", "-m", "initial commit"]);

        // Create a wrapper around pytest that logs every invocation so we can assert
        // exact cache hits vs. invalidations deterministically on Unix and Windows.
        let calls_log = tmp.path().join("pytest_calls.log");
        #[cfg(unix)]
        let wrapper_path = {
            let path = tmp.path().join("pytest_wrapper.sh");
            let script = format!(
                "#!/bin/sh\necho call >> \"{}\"\nexec pytest \"$@\"\n",
                calls_log.display()
            );
            fs::write(&path, script).expect("write pytest wrapper");
            let mut perms = fs::metadata(&path).expect("meta").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&path, perms).expect("chmod +x wrapper");
            path
        };
        #[cfg(windows)]
        let wrapper_path = {
            let path = tmp.path().join("pytest_wrapper.cmd");
            let script = format!(
                "@echo off\r\necho call>>\"{}\"\r\npytest %*\r\nexit /b %ERRORLEVEL%\r\n",
                calls_log.display()
            );
            fs::write(&path, script).expect("write pytest wrapper");
            path
        };

        let escaped_wrapper =
            serde_json::to_string(&wrapper_path.to_string_lossy()).expect("escape wrapper path");
        fs::write(
            tmp.path().join(".failcast.toml"),
            format!("pytest_bin = {escaped_wrapper}\n"),
        )
        .expect("write .failcast.toml");

        let count_pytest_calls = || -> usize {
            fs::read_to_string(&calls_log)
                .map(|s| s.lines().count())
                .unwrap_or(0)
        };

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .arg("init")
            .assert()
            .success();

        let out1 = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["run", "--budget", "5", "--dry-run", "--json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json1: serde_json::Value = serde_json::from_slice(&out1).expect("parse json1");
        assert_eq!(json1["total_discovered_tests"], 4);
        assert_eq!(
            count_pytest_calls(),
            1,
            "first dry-run must invoke pytest once"
        );

        let out2 = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["run", "--budget", "5", "--dry-run", "--json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json2: serde_json::Value = serde_json::from_slice(&out2).expect("parse json2");
        assert_eq!(json2["total_discovered_tests"], 4);
        assert_eq!(
            count_pytest_calls(),
            1,
            "second dry-run must hit SQLite discovery cache without invoking pytest"
        );

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["explain", "tests/test_calc.py::test_add_positive", "--json"])
            .assert()
            .success();
        assert_eq!(
            count_pytest_calls(),
            1,
            "failcast explain must also hit SQLite discovery cache"
        );

        let test_file = tmp.path().join("tests/test_calc.py");
        let mut test_src = fs::read_to_string(&test_file).expect("read test_calc.py");
        test_src.push_str("\n\ndef test_cache_invalidation_check():\n    assert 2 + 2 == 4\n");
        fs::write(&test_file, test_src).expect("update test_calc.py");

        let out3 = cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args(["run", "--budget", "5", "--dry-run", "--json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json3: serde_json::Value = serde_json::from_slice(&out3).expect("parse json3");
        assert_eq!(
            json3["total_discovered_tests"], 5,
            "cache invalidation must pick up newly added test"
        );
        assert_eq!(
            count_pytest_calls(),
            2,
            "test file modification must invalidate cache and re-run pytest"
        );

        cargo_bin_cmd!("failcast")
            .current_dir(tmp.path())
            .args([
                "run",
                "--budget",
                "5",
                "--dry-run",
                "--json",
                "--refresh-tests",
            ])
            .assert()
            .success();
        assert_eq!(
            count_pytest_calls(),
            3,
            "--refresh-tests must force pytest discovery"
        );
    }
}
