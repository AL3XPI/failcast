use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use pts_core::{CommitRecord, DiffFile, DiffSummary, PtsError, normalize_posix_path};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const COMMIT_MARKER: &str = "---PTS-COMMIT---";

/// Returns `true` when `repo_root` is inside a git working tree.
pub fn is_git_repo(repo_root: &Path) -> bool {
    Command::new("git")
        .arg("rev-parse")
        .arg("--is-inside-work-tree")
        .current_dir(repo_root)
        .output()
        .is_ok_and(|out| out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "true")
}

/// Computes a deterministic privacy-preserving SHA-256 hex hash of an author identity.
pub fn hash_author(identity: &str) -> String {
    let normalized = identity.trim().to_ascii_lowercase();
    let digest = Sha256::digest(normalized.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Extracts the working-tree diff (`git diff HEAD --numstat`) with POSIX repo-relative
/// paths and a deterministic `diff_fingerprint`.
pub fn extract_working_diff(repo_root: &Path) -> Result<DiffSummary, PtsError> {
    if !is_git_repo(repo_root) {
        return Err(PtsError::Runner(format!(
            "{} is not a git repository",
            repo_root.display()
        )));
    }

    let base_commit = resolve_head_sha(repo_root)?;
    let mut cmd = Command::new("git");
    cmd.args(["-c", "core.quotepath=false", "diff"]);
    if base_commit.is_some() {
        cmd.arg("HEAD");
    } else {
        cmd.arg("--cached");
    }
    cmd.args(["--numstat", "--no-renames"]).current_dir(repo_root);

    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PtsError::Runner(format!(
            "git diff failed (exit {:?}): {}",
            output.status.code(),
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let files = parse_numstat_block(&stdout);
    Ok(DiffSummary::from_files(base_commit, files))
}

/// Mines up to `limit` commits from `git log` returning commit metadata, hashed author IDs,
/// and per-file line diff statistics.
pub fn mine_commit_history(repo_root: &Path, limit: usize) -> Result<Vec<CommitRecord>, PtsError> {
    mine_commit_history_from_ref(repo_root, "HEAD", limit)
}

/// Mines up to `limit` first-parent commits starting from `git_ref`, returning commit
/// metadata, hashed author IDs, and per-file line diff statistics against each commit's
/// first parent. Returned in newest-first (`git log`) order.
pub fn mine_commit_history_from_ref(
    repo_root: &Path,
    git_ref: &str,
    limit: usize,
) -> Result<Vec<CommitRecord>, PtsError> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    if !is_git_repo(repo_root) {
        return Err(PtsError::Runner(format!(
            "{} is not a git repository",
            repo_root.display()
        )));
    }

    if git_ref == "HEAD" && resolve_head_sha(repo_root)?.is_none() {
        return Ok(Vec::new());
    }

    let format_arg = format!("--format={COMMIT_MARKER}%n%H%x1f%P%x1f%ae%x1f%an%x1f%ct");
    let output = Command::new("git")
        .args([
            "-c",
            "core.quotepath=false",
            "log",
            "-m",
            "--first-parent",
            "-n",
            &limit.to_string(),
            "--no-renames",
            "--numstat",
            &format_arg,
            git_ref,
        ])
        .current_dir(repo_root)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PtsError::Runner(format!(
            "git log failed (exit {:?}): {}",
            output.status.code(),
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_git_log_output(&stdout)
}

/// RAII temporary detached `git worktree` that isolates historical checkouts from the
/// user's working tree and always removes/prunes the worktree on `Drop`.
pub struct TempGitWorktree {
    repo_root: PathBuf,
    worktree_path: PathBuf,
    _tempdir: TempDir,
}

impl TempGitWorktree {
    pub fn create(repo_root: &Path, commit_sha: &str) -> Result<Self, PtsError> {
        if !is_git_repo(repo_root) {
            return Err(PtsError::Runner(format!(
                "{} is not a git repository",
                repo_root.display()
            )));
        }

        let tempdir = tempfile::tempdir()?;
        let worktree_path = tempdir.path().join("worktree");
        let wt_str = worktree_path.to_string_lossy().to_string();

        let output = Command::new("git")
            .args(["worktree", "add", "--detach", &wt_str, commit_sha])
            .current_dir(repo_root)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PtsError::Runner(format!(
                "git worktree add failed for {commit_sha}: {}",
                stderr.trim()
            )));
        }

        Ok(Self {
            repo_root: repo_root.to_path_buf(),
            worktree_path,
            _tempdir: tempdir,
        })
    }

    pub fn path(&self) -> &Path {
        &self.worktree_path
    }

    pub fn checkout(&self, commit_sha: &str) -> Result<(), PtsError> {
        let output = Command::new("git")
            .args(["checkout", "--detach", "--force", commit_sha])
            .current_dir(&self.worktree_path)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PtsError::Runner(format!(
                "git checkout failed in worktree for {commit_sha}: {}",
                stderr.trim()
            )));
        }

        let _ = Command::new("git")
            .args(["clean", "-fd"])
            .current_dir(&self.worktree_path)
            .output();

        Ok(())
    }
}

impl Drop for TempGitWorktree {
    fn drop(&mut self) {
        let wt_str = self.worktree_path.to_string_lossy().to_string();
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force", &wt_str])
            .current_dir(&self.repo_root)
            .output();
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&self.repo_root)
            .output();
    }
}

fn resolve_head_sha(repo_root: &Path) -> Result<Option<String>, PtsError> {
    let output = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(repo_root)
        .output()?;

    if output.status.success() {
        let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if sha.is_empty() {
            Ok(None)
        } else {
            Ok(Some(sha))
        }
    } else {
        // Unborn HEAD (repository initialized with no commits yet).
        Ok(None)
    }
}

fn parse_numstat_line(line: &str) -> Option<DiffFile> {
    let mut parts = line.splitn(3, '\t');
    let added_raw = parts.next()?;
    let deleted_raw = parts.next()?;
    let path_raw = parts.next()?;

    let path = normalize_posix_path(path_raw.trim());
    if path.is_empty() || path == ".pts" || path.starts_with(".pts/") {
        return None;
    }

    let lines_added = if added_raw == "-" {
        0
    } else {
        added_raw.trim().parse::<u32>().ok()?
    };
    let lines_deleted = if deleted_raw == "-" {
        0
    } else {
        deleted_raw.trim().parse::<u32>().ok()?
    };

    Some(DiffFile {
        path,
        lines_added,
        lines_deleted,
    })
}

fn parse_numstat_block(text: &str) -> Vec<DiffFile> {
    let mut by_path: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for line in text.lines() {
        if let Some(df) = parse_numstat_line(line) {
            let entry = by_path.entry(df.path).or_insert((0, 0));
            entry.0 = entry.0.saturating_add(df.lines_added);
            entry.1 = entry.1.saturating_add(df.lines_deleted);
        }
    }

    by_path
        .into_iter()
        .map(|(path, (lines_added, lines_deleted))| DiffFile {
            path,
            lines_added,
            lines_deleted,
        })
        .collect()
}

fn parse_git_log_output(stdout: &str) -> Result<Vec<CommitRecord>, PtsError> {
    let mut commits = Vec::new();
    let mut lines = stdout.lines().peekable();

    while let Some(line) = lines.next() {
        if line.trim() != COMMIT_MARKER {
            continue;
        }
        let Some(header) = lines.next() else {
            break;
        };

        let fields: Vec<&str> = header.split('\x1f').collect();
        if fields.len() < 5 {
            return Err(PtsError::Parse(format!(
                "malformed git log commit header: {header}"
            )));
        }

        let commit_sha = fields[0].trim().to_string();
        let parent_sha = fields[1].split_whitespace().next().map(String::from);
        let author_email = fields[2].trim();
        let author_name = fields[3].trim();
        let author_identity = if !author_email.is_empty() {
            author_email
        } else {
            author_name
        };
        let author_hash = hash_author(author_identity);
        let commit_ts = fields[4].trim().parse::<i64>().map_err(|e| {
            PtsError::Parse(format!("invalid commit timestamp '{}': {e}", fields[4]))
        })?;

        let mut numstat_lines = String::new();
        while let Some(&next_line) = lines.peek() {
            if next_line.trim() == COMMIT_MARKER {
                break;
            }
            let consumed = lines.next().unwrap_or_default();
            if !consumed.trim().is_empty() {
                numstat_lines.push_str(consumed);
                numstat_lines.push('\n');
            }
        }

        let files = parse_numstat_block(&numstat_lines);
        let files_changed = files.len() as u32;
        let lines_added = files
            .iter()
            .fold(0u32, |acc, f| acc.saturating_add(f.lines_added));
        let lines_deleted = files
            .iter()
            .fold(0u32, |acc, f| acc.saturating_add(f.lines_deleted));

        commits.push(CommitRecord {
            commit_sha,
            parent_sha,
            author_hash,
            commit_ts,
            files_changed,
            lines_added,
            lines_deleted,
            files,
        });
    }

    Ok(commits)
}
