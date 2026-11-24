pub mod config;
pub mod error;
pub mod runner;
pub mod types;

pub use config::Config;
pub use error::PtsError;
pub use runner::TestRunner;
pub use types::{
    DiffFile, DiffSummary, RunMode, RunnerResult, SelectionReason, TestCase, TestExecutionOutcome,
    TestStatus,
};

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_toml_override() {
        let dir = tempfile::tempdir().expect("tempdir");

        // When .pts.toml is absent, defaults are returned.
        let default_cfg = Config::load(dir.path()).expect("load default config");
        assert_eq!(default_cfg, Config::default());
        assert_eq!(default_cfg.runner, "pytest");
        assert_eq!(default_cfg.default_budget_secs, 15);
        assert!((default_cfg.explore_fraction - 0.10).abs() < f64::EPSILON);
        assert_eq!(default_cfg.min_failures_for_ml, 25);
        assert_eq!(default_cfg.min_runs_for_ml, 100);
        assert_eq!(default_cfg.history_commit_limit, 500);
        assert_eq!(default_cfg.pytest_bin, "pytest");
        assert_eq!(default_cfg.max_traceback_chars, 1500);

        // Partial TOML overrides only specified keys.
        let toml_content = r#"
default_budget_secs = 42
pytest_bin = "/usr/local/bin/pytest"
max_traceback_chars = 800
"#;
        std::fs::write(dir.path().join(".pts.toml"), toml_content).expect("write .pts.toml");

        let overridden = Config::load(dir.path()).expect("load overridden config");
        assert_eq!(overridden.default_budget_secs, 42);
        assert_eq!(overridden.pytest_bin, "/usr/local/bin/pytest");
        assert_eq!(overridden.max_traceback_chars, 800);
        // Unspecified keys remain defaults.
        assert_eq!(overridden.runner, "pytest");
        assert!((overridden.explore_fraction - 0.10).abs() < f64::EPSILON);
        assert_eq!(overridden.min_failures_for_ml, 25);
        assert_eq!(overridden.min_runs_for_ml, 100);
        assert_eq!(overridden.history_commit_limit, 500);
    }

    #[test]
    fn diff_fingerprint_is_order_independent() {
        let files_a = vec![
            DiffFile {
                path: "src/b.py".into(),
                lines_added: 3,
                lines_deleted: 1,
            },
            DiffFile {
                path: "src/a.py".into(),
                lines_added: 10,
                lines_deleted: 0,
            },
        ];
        let files_b = vec![files_a[1].clone(), files_a[0].clone()];
        assert_eq!(
            DiffSummary::compute_fingerprint(&files_a),
            DiffSummary::compute_fingerprint(&files_b)
        );
    }
}
