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
}
