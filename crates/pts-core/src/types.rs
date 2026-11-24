use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::PtsError;

/// Normalizes a repo-relative path to POSIX forward-slash form without leading `./` or `/`.
pub fn normalize_posix_path(raw: &str) -> String {
    let mut path = raw.replace('\\', "/");
    while let Some(stripped) = path.strip_prefix("./") {
        path = stripped.to_string();
    }
    while let Some(stripped) = path.strip_prefix('/') {
        path = stripped.to_string();
    }
    path
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    Pass,
    Fail,
    Error,
    Skip,
}

impl TestStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Error => "error",
            Self::Skip => "skip",
        }
    }
}

impl fmt::Display for TestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TestStatus {
    type Err = PtsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pass" => Ok(Self::Pass),
            "fail" => Ok(Self::Fail),
            "error" => Ok(Self::Error),
            "skip" => Ok(Self::Skip),
            other => Err(PtsError::Parse(format!("invalid test status: {other}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionReason {
    Full,
    Heuristic,
    Ranked,
    Explore,
}

impl SelectionReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Heuristic => "heuristic",
            Self::Ranked => "ranked",
            Self::Explore => "explore",
        }
    }
}

impl fmt::Display for SelectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SelectionReason {
    type Err = PtsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "heuristic" => Ok(Self::Heuristic),
            "ranked" => Ok(Self::Ranked),
            "explore" => Ok(Self::Explore),
            other => Err(PtsError::Parse(format!(
                "invalid selection reason: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    All,
    Budget,
    TopK,
    Backfill,
}

impl RunMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Budget => "budget",
            Self::TopK => "top_k",
            Self::Backfill => "backfill",
        }
    }
}

impl fmt::Display for RunMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RunMode {
    type Err = PtsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "all" => Ok(Self::All),
            "budget" => Ok(Self::Budget),
            "top_k" => Ok(Self::TopK),
            "backfill" => Ok(Self::Backfill),
            other => Err(PtsError::Parse(format!("invalid run mode: {other}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffFile {
    /// Repo-relative POSIX path (e.g. "src/auth/token.py").
    pub path: String,
    pub lines_added: u32,
    pub lines_deleted: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffSummary {
    pub base_commit: Option<String>,
    /// SHA-256 hex digest of canonical sorted diff paths + line counts.
    pub diff_fingerprint: String,
    pub files: Vec<DiffFile>,
}

impl DiffSummary {
    pub fn compute_fingerprint(files: &[DiffFile]) -> String {
        let mut sorted: Vec<&DiffFile> = files.iter().collect();
        sorted.sort_by(|a, b| a.path.cmp(&b.path));

        let mut hasher = Sha256::new();
        for f in sorted {
            hasher.update(f.path.as_bytes());
            hasher.update(b"\0");
            hasher.update(f.lines_added.to_le_bytes());
            hasher.update(f.lines_deleted.to_le_bytes());
            hasher.update(b"\n");
        }
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use fmt::Write as _;
            let _ = write!(&mut hex, "{byte:02x}");
        }
        hex
    }

    pub fn from_files(base_commit: Option<String>, mut files: Vec<DiffFile>) -> Self {
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let diff_fingerprint = Self::compute_fingerprint(&files);
        Self {
            base_commit,
            diff_fingerprint,
            files,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestCase {
    /// Canonical runner ID, e.g. "tests/test_auth.py::test_expired".
    pub test_id: String,
    /// Repo-relative POSIX path ("tests/test_auth.py").
    pub file_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestExecutionOutcome {
    pub test_id: String,
    pub status: TestStatus,
    pub duration_ms: u64,
    /// Truncated to `max_traceback_chars` (default 1,500 chars max).
    pub failure_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunnerResult {
    pub exit_code: i32,
    pub total_duration_ms: u64,
    pub outcomes: Vec<TestExecutionOutcome>,
}

/// A mined git commit with its per-file line diff statistics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRecord {
    pub commit_sha: String,
    pub parent_sha: Option<String>,
    /// Privacy-preserving SHA-256 hex digest of the commit author identity.
    pub author_hash: String,
    pub commit_ts: i64,
    pub files_changed: u32,
    pub lines_added: u32,
    pub lines_deleted: u32,
    pub files: Vec<DiffFile>,
}

/// Historical test-level statistics strictly prior to a snapshot timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestHistory {
    pub test_id: String,
    pub total_runs: u64,
    pub total_failures: u64,
    pub ewma_fail_10: f64,
    pub ewma_fail_50: f64,
    pub runs_since_last_fail: Option<u64>,
    pub median_duration_ms: Option<f64>,
    pub flakiness_rate: f64,
    pub last_run_ts: Option<i64>,
}

/// Co-failure counts observed when `test_id` ran against a diff touching `file_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CoFailCounts {
    pub co_fails: u64,
    pub co_runs: u64,
}

/// Pure, read-only snapshot of repository history strictly prior to `as_of_ts`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HistorySnapshot {
    pub as_of_ts: i64,
    pub total_runs: usize,
    pub total_failures: usize,
    pub median_repo_test_ms: Option<f64>,
    pub test_history: BTreeMap<String, TestHistory>,
    /// Keyed by `(test_id, file_path)`.
    pub cofail_counts: BTreeMap<(String, String), CoFailCounts>,
    /// Total commits touching `file_path` with `commit_ts < as_of_ts`.
    pub file_commit_counts: BTreeMap<String, u32>,
    /// Pairwise commit co-occurrence counts keyed by ordered `(min_path, max_path)`.
    pub pair_commit_counts: BTreeMap<(String, String), u32>,
    /// 30-day commit count per `file_path` in `[as_of_ts - 30d, as_of_ts)`.
    pub file_churn_30d: BTreeMap<String, u32>,
}

impl HistorySnapshot {
    /// Laplace-smoothed historical P(test fails | file in diff):
    /// `(cofails + 0.5) / (co_runs + 5.0)`.
    pub fn cofail_rate(&self, test_id: &str, file_path: &str) -> f64 {
        let counts = self
            .cofail_counts
            .get(&(test_id.to_string(), file_path.to_string()))
            .copied()
            .unwrap_or_default();
        (counts.co_fails as f64 + 0.5) / (counts.co_runs as f64 + 5.0)
    }

    /// Git commit co-occurrence ratio between `test_file` and `modified_file`:
    /// `commits(test_file & modified_file) / commits(modified_file)`.
    pub fn cochange_freq(&self, test_file: &str, modified_file: &str) -> f64 {
        let Some(&mod_count) = self.file_commit_counts.get(modified_file) else {
            return 0.0;
        };
        if mod_count == 0 {
            return 0.0;
        }
        if test_file == modified_file {
            return 1.0;
        }
        let key = if test_file <= modified_file {
            (test_file.to_string(), modified_file.to_string())
        } else {
            (modified_file.to_string(), test_file.to_string())
        };
        let pair_count = self.pair_commit_counts.get(&key).copied().unwrap_or(0);
        (pair_count as f64 / mod_count as f64).min(1.0)
    }

    /// 30-day commit count for `file_path` strictly before `as_of_ts`.
    pub fn churn_30d(&self, file_path: &str) -> u32 {
        self.file_churn_30d.get(file_path).copied().unwrap_or(0)
    }
}

/// A single persisted test outcome within a historical run, used for pure ML
/// feature accumulation and walk-forward evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalOutcome {
    pub test_id: String,
    pub file_path: String,
    pub status: TestStatus,
    pub duration_ms: u64,
    pub sampling_propensity: f64,
    /// Optional precomputed directed AST hop distance from `file_path` to any modified file.
    pub ast_min_hop: Option<usize>,
}

/// A persisted test run with its diff files and test outcomes, ordered chronologically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalRun {
    pub run_id: String,
    pub commit_sha: Option<String>,
    pub diff_fingerprint: String,
    pub started_at: i64,
    pub diff_files: Vec<DiffFile>,
    pub outcomes: Vec<HistoricalOutcome>,
}
