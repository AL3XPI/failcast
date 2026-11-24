use std::path::Path;
use std::process::Command;
use std::time::Instant;

use pts_core::{Config, PtsError, RunnerResult, TestCase, TestRunner};
use tracing::debug;

use crate::junit::{normalize_posix_path, parse_junit_xml};

#[derive(Debug, Clone)]
pub struct PytestRunner {
    pytest_bin: String,
    max_traceback_chars: usize,
    extra_env: Vec<(String, String)>,
}

impl Default for PytestRunner {
    fn default() -> Self {
        let cfg = Config::default();
        Self {
            pytest_bin: cfg.pytest_bin,
            max_traceback_chars: cfg.max_traceback_chars,
            extra_env: Vec::new(),
        }
    }
}

impl PytestRunner {
    pub fn new(pytest_bin: impl Into<String>, max_traceback_chars: usize) -> Self {
        Self {
            pytest_bin: pytest_bin.into(),
            max_traceback_chars,
            extra_env: Vec::new(),
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self::new(config.pytest_bin.clone(), config.max_traceback_chars)
    }

    /// Attaches an extra environment variable for spawned pytest subprocesses.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_env.push((key.into(), value.into()));
        self
    }

    fn base_command(&self, repo_root: &Path) -> Command {
        let mut cmd = Command::new(&self.pytest_bin);
        cmd.current_dir(repo_root);
        for (k, v) in &self.extra_env {
            cmd.env(k, v);
        }
        cmd
    }
}

impl TestRunner for PytestRunner {
    fn name(&self) -> &'static str {
        "pytest"
    }

    fn discover(&self, repo_root: &Path) -> Result<Vec<TestCase>, PtsError> {
        let output = self
            .base_command(repo_root)
            .args(["--collect-only", "-q", "--no-header"])
            .output()
            .map_err(|e| {
                PtsError::Runner(format!(
                    "failed to spawn `{}` for discovery: {e}",
                    self.pytest_bin
                ))
            })?;

        let exit_code = output.status.code().unwrap_or(-1);
        // pytest exit code 5 means no tests collected.
        if exit_code == 5 {
            return Ok(Vec::new());
        }
        if exit_code != 0 {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(PtsError::Runner(format!(
                "pytest discovery failed (exit {exit_code}): {stderr}\n{stdout}"
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut cases = Vec::new();
        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || !trimmed.contains("::") {
                continue;
            }
            if let Some((raw_file, rest)) = trimmed.split_once("::") {
                let file_path = normalize_posix_path(raw_file);
                let test_id = format!("{file_path}::{rest}");
                cases.push(TestCase { test_id, file_path });
            }
        }

        debug!(count = cases.len(), "discovered pytest test cases");
        Ok(cases)
    }

    fn execute(
        &self,
        repo_root: &Path,
        selected_ids: Option<&[String]>,
    ) -> Result<RunnerResult, PtsError> {
        if let Some(ids) = selected_ids
            && ids.is_empty()
        {
            return Ok(RunnerResult {
                exit_code: 0,
                total_duration_ms: 0,
                outcomes: Vec::new(),
            });
        }

        let xml_dir = tempfile::tempdir()?;
        let xml_path = xml_dir.path().join("junit.xml");

        let mut cmd = self.base_command(repo_root);
        cmd.arg(format!("--junitxml={}", xml_path.display()))
            .args(["-o", "junit_family=legacy", "-q"]);

        if let Some(ids) = selected_ids {
            cmd.args(ids);
        }

        let start = Instant::now();
        let output = cmd.output().map_err(|e| {
            PtsError::Runner(format!(
                "failed to spawn `{}` for execution: {e}",
                self.pytest_bin
            ))
        })?;
        let total_duration_ms = start.elapsed().as_millis() as u64;
        let exit_code = output.status.code().unwrap_or(-1);

        let outcomes = if xml_path.is_file() {
            let xml = std::fs::read_to_string(&xml_path)?;
            parse_junit_xml(&xml, self.max_traceback_chars)?
        } else if exit_code == 5 || exit_code == 0 {
            Vec::new()
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(PtsError::Runner(format!(
                "pytest exited with code {exit_code} without producing JUnit XML:\nstderr: {stderr}\nstdout: {stdout}"
            )));
        };

        debug!(
            exit_code,
            total_duration_ms,
            outcomes = outcomes.len(),
            "pytest execution completed"
        );

        Ok(RunnerResult {
            exit_code,
            total_duration_ms,
            outcomes,
        })
    }
}
