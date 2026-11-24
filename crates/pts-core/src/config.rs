use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::PtsError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub runner: String,
    pub default_budget_secs: u64,
    pub explore_fraction: f64,
    pub min_failures_for_ml: usize,
    pub min_runs_for_ml: usize,
    pub history_commit_limit: usize,
    pub pytest_bin: String,
    pub max_traceback_chars: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            runner: "pytest".to_string(),
            default_budget_secs: 15,
            explore_fraction: 0.10,
            min_failures_for_ml: 25,
            min_runs_for_ml: 100,
            history_commit_limit: 500,
            pytest_bin: "pytest".to_string(),
            max_traceback_chars: 1500,
        }
    }
}

impl Config {
    pub fn from_toml_str(content: &str) -> Result<Self, PtsError> {
        toml::from_str::<Config>(content)
            .map_err(|e| PtsError::Config(format!("failed to parse config TOML: {e}")))
    }

    /// Loads configuration from `<repo_root>/.pts.toml` (falling back to
    /// `<repo_root>/.pts/config.toml`), or returns defaults when neither exists.
    pub fn load(repo_root: &Path) -> Result<Self, PtsError> {
        let root_cfg = repo_root.join(".pts.toml");
        if root_cfg.is_file() {
            let content = std::fs::read_to_string(&root_cfg)?;
            return Self::from_toml_str(&content);
        }

        let state_cfg = repo_root.join(".pts").join("config.toml");
        if state_cfg.is_file() {
            let content = std::fs::read_to_string(&state_cfg)?;
            return Self::from_toml_str(&content);
        }

        Ok(Self::default())
    }
}
