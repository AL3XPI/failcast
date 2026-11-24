use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::PtsError;

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
            other => Err(PtsError::Parse(format!("invalid selection reason: {other}"))),
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
