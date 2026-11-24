mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use assert_cmd::cargo::cargo_bin_cmd;
    use rusqlite::Connection;

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
}
