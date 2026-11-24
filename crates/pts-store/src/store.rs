use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use pts_core::{
    CoFailCounts, CommitRecord, DiffFile, HistoricalOutcome, HistoricalRun, HistorySnapshot,
    PtsError, RunMode, SelectionReason, TestCase, TestHistory, TestStatus,
};
use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA_SQL: &str = include_str!("schema.sql");
const SCHEMA_VERSION: i64 = 1;
const SECONDS_PER_30_DAYS: i64 = 30 * 86_400;
const MAX_COCHANGE_FILES_PER_COMMIT: usize = 200;
const EWMA_ALPHA_10: f64 = 0.2;
const EWMA_ALPHA_50: f64 = 0.04;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRecord {
    pub run_id: String,
    pub commit_sha: Option<String>,
    pub diff_fingerprint: String,
    pub runner: String,
    pub mode: RunMode,
    pub budget_ms: Option<i64>,
    pub started_at: i64,
    pub finished_at: Option<i64>,
    pub total_duration_ms: Option<i64>,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestOutcomeRecord {
    pub run_id: String,
    pub test_id: String,
    pub status: TestStatus,
    pub duration_ms: u64,
    pub predicted_prob: Option<f64>,
    pub selection_reason: SelectionReason,
    pub sampling_propensity: f64,
    pub failure_excerpt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestRecord {
    pub test_id: String,
    pub file_path: String,
    pub first_seen_ts: i64,
    pub last_seen_ts: i64,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelRecord {
    pub model_id: String,
    pub trained_at: i64,
    pub max_train_ts: i64,
    pub train_rows: usize,
    pub pos_rows: usize,
    pub brier_score: f64,
    pub calibration_a: f64,
    pub calibration_b: f64,
    pub artifact_path: String,
}

/// Complete run payload for atomic multi-run persistence via `Store::persist_runs_batch`.
#[derive(Debug, Clone)]
pub struct RunBatchEntry {
    pub run: RunRecord,
    pub diff_files: Vec<DiffFile>,
    pub tests: Vec<TestCase>,
    pub outcomes: Vec<TestOutcomeRecord>,
    pub seen_ts: i64,
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PtsError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path).map_err(store_err)?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA synchronous = NORMAL;
            PRAGMA busy_timeout = 5000;
            ",
        )
        .map_err(store_err)?;

        Ok(Self { conn })
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    pub fn migrate(&self) -> Result<(), PtsError> {
        self.conn.execute_batch(SCHEMA_SQL).map_err(store_err)?;

        let applied_at = current_unix_ts();
        self.conn
            .execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                params![SCHEMA_VERSION, applied_at],
            )
            .map_err(store_err)?;

        Ok(())
    }

    pub fn upsert_test(&self, test: &TestCase, seen_ts: i64) -> Result<(), PtsError> {
        self.conn
            .execute(
                "
                INSERT INTO tests (test_id, file_path, first_seen_ts, last_seen_ts, active)
                VALUES (?1, ?2, ?3, ?3, 1)
                ON CONFLICT(test_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    last_seen_ts = MAX(tests.last_seen_ts, excluded.last_seen_ts),
                    active = 1
                ",
                params![test.test_id, test.file_path, seen_ts],
            )
            .map_err(store_err)?;
        Ok(())
    }

    pub fn insert_run(&self, run: &RunRecord) -> Result<(), PtsError> {
        self.conn
            .execute(
                "
                INSERT INTO runs (
                    run_id, commit_sha, diff_fingerprint, runner, mode,
                    budget_ms, started_at, finished_at, total_duration_ms, exit_code
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ",
                params![
                    run.run_id,
                    run.commit_sha,
                    run.diff_fingerprint,
                    run.runner,
                    run.mode.as_str(),
                    run.budget_ms,
                    run.started_at,
                    run.finished_at,
                    run.total_duration_ms,
                    run.exit_code,
                ],
            )
            .map_err(store_err)?;
        Ok(())
    }

    pub fn finish_run(
        &self,
        run_id: &str,
        finished_at: i64,
        total_duration_ms: u64,
        exit_code: i32,
    ) -> Result<(), PtsError> {
        self.conn
            .execute(
                "
                UPDATE runs
                SET finished_at = ?2, total_duration_ms = ?3, exit_code = ?4
                WHERE run_id = ?1
                ",
                params![run_id, finished_at, total_duration_ms as i64, exit_code],
            )
            .map_err(store_err)?;
        Ok(())
    }

    pub fn insert_run_diff_files(
        &self,
        run_id: &str,
        files: &[DiffFile],
    ) -> Result<(), PtsError> {
        let mut stmt = self
            .conn
            .prepare(
                "
                INSERT OR REPLACE INTO run_diff_files (run_id, file_path, lines_added, lines_deleted)
                VALUES (?1, ?2, ?3, ?4)
                ",
            )
            .map_err(store_err)?;

        for file in files {
            stmt.execute(params![
                run_id,
                file.path,
                file.lines_added as i64,
                file.lines_deleted as i64,
            ])
            .map_err(store_err)?;
        }
        Ok(())
    }

    pub fn insert_test_outcome(&self, outcome: &TestOutcomeRecord) -> Result<(), PtsError> {
        self.conn
            .execute(
                "
                INSERT OR REPLACE INTO test_outcomes (
                    run_id, test_id, status, duration_ms, predicted_prob,
                    selection_reason, sampling_propensity, failure_excerpt
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                ",
                params![
                    outcome.run_id,
                    outcome.test_id,
                    outcome.status.as_str(),
                    outcome.duration_ms as i64,
                    outcome.predicted_prob,
                    outcome.selection_reason.as_str(),
                    outcome.sampling_propensity,
                    outcome.failure_excerpt,
                ],
            )
            .map_err(store_err)?;
        Ok(())
    }

    pub fn persist_run_with_outcomes(
        &mut self,
        run: &RunRecord,
        diff_files: &[DiffFile],
        tests: &[TestCase],
        outcomes: &[TestOutcomeRecord],
        seen_ts: i64,
    ) -> Result<(), PtsError> {
        let tx = self.conn.transaction().map_err(store_err)?;

        tx.execute(
            "
            INSERT INTO runs (
                run_id, commit_sha, diff_fingerprint, runner, mode,
                budget_ms, started_at, finished_at, total_duration_ms, exit_code
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                run.run_id,
                run.commit_sha,
                run.diff_fingerprint,
                run.runner,
                run.mode.as_str(),
                run.budget_ms,
                run.started_at,
                run.finished_at,
                run.total_duration_ms,
                run.exit_code,
            ],
        )
        .map_err(store_err)?;

        {
            let mut diff_stmt = tx
                .prepare(
                    "
                    INSERT OR REPLACE INTO run_diff_files (run_id, file_path, lines_added, lines_deleted)
                    VALUES (?1, ?2, ?3, ?4)
                    ",
                )
                .map_err(store_err)?;
            for file in diff_files {
                diff_stmt
                    .execute(params![
                        run.run_id,
                        file.path,
                        file.lines_added as i64,
                        file.lines_deleted as i64,
                    ])
                    .map_err(store_err)?;
            }
        }

        {
            let mut test_stmt = tx
                .prepare(
                    "
                    INSERT INTO tests (test_id, file_path, first_seen_ts, last_seen_ts, active)
                    VALUES (?1, ?2, ?3, ?3, 1)
                    ON CONFLICT(test_id) DO UPDATE SET
                        file_path = excluded.file_path,
                        last_seen_ts = MAX(tests.last_seen_ts, excluded.last_seen_ts),
                        active = 1
                    ",
                )
                .map_err(store_err)?;
            for test in tests {
                test_stmt
                    .execute(params![test.test_id, test.file_path, seen_ts])
                    .map_err(store_err)?;
            }
        }

        {
            let mut outcome_stmt = tx
                .prepare(
                    "
                    INSERT OR REPLACE INTO test_outcomes (
                        run_id, test_id, status, duration_ms, predicted_prob,
                        selection_reason, sampling_propensity, failure_excerpt
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                    ",
                )
                .map_err(store_err)?;
            for outcome in outcomes {
                outcome_stmt
                    .execute(params![
                        outcome.run_id,
                        outcome.test_id,
                        outcome.status.as_str(),
                        outcome.duration_ms as i64,
                        outcome.predicted_prob,
                        outcome.selection_reason.as_str(),
                        outcome.sampling_propensity,
                        outcome.failure_excerpt,
                    ])
                    .map_err(store_err)?;
            }
        }

        tx.commit().map_err(store_err)?;
        Ok(())
    }

    /// Persists multiple runs, their diff files, tests, and outcomes inside a single atomic
    /// SQLite transaction. Used by `pts import-runs` for fast, all-or-nothing JSONL ingestion.
    pub fn persist_runs_batch(&mut self, runs: &[RunBatchEntry]) -> Result<usize, PtsError> {
        if runs.is_empty() {
            return Ok(0);
        }

        let tx = self.conn.transaction().map_err(store_err)?;

        {
            let mut run_stmt = tx
                .prepare(
                    "
                    INSERT INTO runs (
                        run_id, commit_sha, diff_fingerprint, runner, mode,
                        budget_ms, started_at, finished_at, total_duration_ms, exit_code
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                    ",
                )
                .map_err(store_err)?;

            let mut diff_stmt = tx
                .prepare(
                    "
                    INSERT OR REPLACE INTO run_diff_files (run_id, file_path, lines_added, lines_deleted)
                    VALUES (?1, ?2, ?3, ?4)
                    ",
                )
                .map_err(store_err)?;

            let mut test_stmt = tx
                .prepare(
                    "
                    INSERT INTO tests (test_id, file_path, first_seen_ts, last_seen_ts, active)
                    VALUES (?1, ?2, ?3, ?3, 1)
                    ON CONFLICT(test_id) DO UPDATE SET
                        file_path = excluded.file_path,
                        last_seen_ts = MAX(tests.last_seen_ts, excluded.last_seen_ts),
                        active = 1
                    ",
                )
                .map_err(store_err)?;

            let mut outcome_stmt = tx
                .prepare(
                    "
                    INSERT OR REPLACE INTO test_outcomes (
                        run_id, test_id, status, duration_ms, predicted_prob,
                        selection_reason, sampling_propensity, failure_excerpt
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                    ",
                )
                .map_err(store_err)?;

            for entry in runs {
                let run = &entry.run;
                run_stmt
                    .execute(params![
                        run.run_id,
                        run.commit_sha,
                        run.diff_fingerprint,
                        run.runner,
                        run.mode.as_str(),
                        run.budget_ms,
                        run.started_at,
                        run.finished_at,
                        run.total_duration_ms,
                        run.exit_code,
                    ])
                    .map_err(store_err)?;

                for file in &entry.diff_files {
                    diff_stmt
                        .execute(params![
                            run.run_id,
                            file.path,
                            file.lines_added as i64,
                            file.lines_deleted as i64,
                        ])
                        .map_err(store_err)?;
                }

                for test in &entry.tests {
                    test_stmt
                        .execute(params![test.test_id, test.file_path, entry.seen_ts])
                        .map_err(store_err)?;
                }

                for outcome in &entry.outcomes {
                    outcome_stmt
                        .execute(params![
                            outcome.run_id,
                            outcome.test_id,
                            outcome.status.as_str(),
                            outcome.duration_ms as i64,
                            outcome.predicted_prob,
                            outcome.selection_reason.as_str(),
                            outcome.sampling_propensity,
                            outcome.failure_excerpt,
                        ])
                        .map_err(store_err)?;
                }
            }
        }

        tx.commit().map_err(store_err)?;
        Ok(runs.len())
    }

    pub fn get_run(&self, run_id: &str) -> Result<Option<RunRecord>, PtsError> {
        let row = self
            .conn
            .query_row(
                "
                SELECT run_id, commit_sha, diff_fingerprint, runner, mode,
                       budget_ms, started_at, finished_at, total_duration_ms, exit_code
                FROM runs
                WHERE run_id = ?1
                ",
                params![run_id],
                |r| {
                    let mode_str: String = r.get(4)?;
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        mode_str,
                        r.get::<_, Option<i64>>(5)?,
                        r.get::<_, i64>(6)?,
                        r.get::<_, Option<i64>>(7)?,
                        r.get::<_, Option<i64>>(8)?,
                        r.get::<_, Option<i32>>(9)?,
                    ))
                },
            )
            .optional()
            .map_err(store_err)?;

        match row {
            Some((
                run_id,
                commit_sha,
                diff_fingerprint,
                runner,
                mode_str,
                budget_ms,
                started_at,
                finished_at,
                total_duration_ms,
                exit_code,
            )) => Ok(Some(RunRecord {
                run_id,
                commit_sha,
                diff_fingerprint,
                runner,
                mode: RunMode::from_str(&mode_str)?,
                budget_ms,
                started_at,
                finished_at,
                total_duration_ms,
                exit_code,
            })),
            None => Ok(None),
        }
    }

    pub fn get_outcomes_by_run(&self, run_id: &str) -> Result<Vec<TestOutcomeRecord>, PtsError> {
        let mut stmt = self
            .conn
            .prepare(
                "
                SELECT run_id, test_id, status, duration_ms, predicted_prob,
                       selection_reason, sampling_propensity, failure_excerpt
                FROM test_outcomes
                WHERE run_id = ?1
                ORDER BY test_id ASC
                ",
            )
            .map_err(store_err)?;

        let rows = stmt
            .query_map(params![run_id], map_outcome_row)
            .map_err(store_err)?;

        collect_outcomes(rows)
    }

    pub fn get_outcomes_by_test(&self, test_id: &str) -> Result<Vec<TestOutcomeRecord>, PtsError> {
        let mut stmt = self
            .conn
            .prepare(
                "
                SELECT o.run_id, o.test_id, o.status, o.duration_ms, o.predicted_prob,
                       o.selection_reason, o.sampling_propensity, o.failure_excerpt
                FROM test_outcomes o
                JOIN runs r ON r.run_id = o.run_id
                WHERE o.test_id = ?1
                ORDER BY r.started_at ASC, o.run_id ASC
                ",
            )
            .map_err(store_err)?;

        let rows = stmt
            .query_map(params![test_id], map_outcome_row)
            .map_err(store_err)?;

        collect_outcomes(rows)
    }

    pub fn get_run_diff_files(&self, run_id: &str) -> Result<Vec<DiffFile>, PtsError> {
        let mut stmt = self
            .conn
            .prepare(
                "
                SELECT file_path, lines_added, lines_deleted
                FROM run_diff_files
                WHERE run_id = ?1
                ORDER BY file_path ASC
                ",
            )
            .map_err(store_err)?;

        let rows = stmt
            .query_map(params![run_id], |r| {
                Ok(DiffFile {
                    path: r.get(0)?,
                    lines_added: r.get::<_, i64>(1)? as u32,
                    lines_deleted: r.get::<_, i64>(2)? as u32,
                })
            })
            .map_err(store_err)?;

        let mut files = Vec::new();
        for row in rows {
            files.push(row.map_err(store_err)?);
        }
        Ok(files)
    }

    /// Idempotently ingests mined git commits and their per-file diff statistics.
    pub fn upsert_commits(&mut self, commits: &[CommitRecord]) -> Result<usize, PtsError> {
        let tx = self.conn.transaction().map_err(store_err)?;

        {
            let mut commit_stmt = tx
                .prepare(
                    "
                    INSERT INTO commits (
                        commit_sha, parent_sha, author_hash, commit_ts,
                        files_changed, lines_added, lines_deleted
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                    ON CONFLICT(commit_sha) DO UPDATE SET
                        parent_sha = excluded.parent_sha,
                        author_hash = excluded.author_hash,
                        commit_ts = excluded.commit_ts,
                        files_changed = excluded.files_changed,
                        lines_added = excluded.lines_added,
                        lines_deleted = excluded.lines_deleted
                    ",
                )
                .map_err(store_err)?;

            let mut file_stmt = tx
                .prepare(
                    "
                    INSERT INTO commit_files (commit_sha, file_path, lines_added, lines_deleted)
                    VALUES (?1, ?2, ?3, ?4)
                    ON CONFLICT(commit_sha, file_path) DO UPDATE SET
                        lines_added = excluded.lines_added,
                        lines_deleted = excluded.lines_deleted
                    ",
                )
                .map_err(store_err)?;

            for commit in commits {
                commit_stmt
                    .execute(params![
                        commit.commit_sha,
                        commit.parent_sha,
                        commit.author_hash,
                        commit.commit_ts,
                        commit.files_changed as i64,
                        commit.lines_added as i64,
                        commit.lines_deleted as i64,
                    ])
                    .map_err(store_err)?;

                for file in &commit.files {
                    file_stmt
                        .execute(params![
                            commit.commit_sha,
                            file.path,
                            file.lines_added as i64,
                            file.lines_deleted as i64,
                        ])
                        .map_err(store_err)?;
                }
            }
        }

        tx.commit().map_err(store_err)?;
        Ok(commits.len())
    }

    pub fn commit_count(&self) -> Result<usize, PtsError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM commits", [], |r| r.get(0))
            .map_err(store_err)?;
        Ok(count.max(0) as usize)
    }

    pub fn run_count(&self) -> Result<usize, PtsError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM runs", [], |r| r.get(0))
            .map_err(store_err)?;
        Ok(count.max(0) as usize)
    }

    /// Read-only point-in-time history query strictly excluding any run or commit
    /// with timestamp `>= as_of_ts` (anti-leakage invariant).
    pub fn load_history_snapshot(&self, as_of_ts: i64) -> Result<HistorySnapshot, PtsError> {
        let total_runs: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE started_at < ?1",
                params![as_of_ts],
                |r| r.get(0),
            )
            .map_err(store_err)?;

        let mut snapshot = HistorySnapshot {
            as_of_ts,
            total_runs: total_runs.max(0) as usize,
            ..HistorySnapshot::default()
        };

        // 1. Chronological per-test outcomes strictly prior to as_of_ts.
        struct TestAcc {
            total_runs: u64,
            total_failures: u64,
            ewma_10: f64,
            ewma_50: f64,
            runs_since_last_fail: Option<u64>,
            durations: Vec<u64>,
            last_run_ts: Option<i64>,
            fp_flags: BTreeMap<String, (bool, bool)>, // (has_pass, has_fail)
        }

        let mut accs: BTreeMap<String, TestAcc> = BTreeMap::new();
        let mut repo_durations: Vec<u64> = Vec::new();

        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT o.test_id, o.status, o.duration_ms, r.started_at, r.diff_fingerprint
                    FROM test_outcomes o
                    JOIN runs r ON r.run_id = o.run_id
                    WHERE r.started_at < ?1
                    ORDER BY r.started_at ASC, r.run_id ASC, o.test_id ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map(params![as_of_ts], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                })
                .map_err(store_err)?;

            for row in rows {
                let (test_id, status_str, duration_ms, started_at, diff_fp) =
                    row.map_err(store_err)?;
                let status = TestStatus::from_str(&status_str)?;
                let dur = duration_ms.max(0) as u64;

                let entry = accs.entry(test_id).or_insert_with(|| TestAcc {
                    total_runs: 0,
                    total_failures: 0,
                    ewma_10: 0.0,
                    ewma_50: 0.0,
                    runs_since_last_fail: None,
                    durations: Vec::new(),
                    last_run_ts: None,
                    fp_flags: BTreeMap::new(),
                });

                entry.total_runs += 1;
                entry.last_run_ts = Some(match entry.last_run_ts {
                    Some(prev) => prev.max(started_at),
                    None => started_at,
                });

                match status {
                    TestStatus::Fail | TestStatus::Error => {
                        entry.total_failures += 1;
                        snapshot.total_failures += 1;
                        entry.ewma_10 = EWMA_ALPHA_10 + (1.0 - EWMA_ALPHA_10) * entry.ewma_10;
                        entry.ewma_50 = EWMA_ALPHA_50 + (1.0 - EWMA_ALPHA_50) * entry.ewma_50;
                        entry.runs_since_last_fail = Some(0);
                        entry.durations.push(dur);
                        repo_durations.push(dur);
                        entry.fp_flags.entry(diff_fp).or_insert((false, false)).1 = true;
                    }
                    TestStatus::Pass => {
                        entry.ewma_10 *= 1.0 - EWMA_ALPHA_10;
                        entry.ewma_50 *= 1.0 - EWMA_ALPHA_50;
                        if let Some(since) = entry.runs_since_last_fail {
                            entry.runs_since_last_fail = Some(since + 1);
                        }
                        entry.durations.push(dur);
                        repo_durations.push(dur);
                        entry.fp_flags.entry(diff_fp).or_insert((false, false)).0 = true;
                    }
                    TestStatus::Skip => {
                        // Skipped runs update total_runs and last_run_ts but do not bias
                        // failure rates, durations, or flakiness.
                    }
                }
            }
        }

        snapshot.median_repo_test_ms = median_u64(&mut repo_durations);

        for (test_id, mut acc) in accs {
            let total_fps = acc.fp_flags.len();
            let flaky_fps = acc
                .fp_flags
                .values()
                .filter(|&&(has_pass, has_fail)| has_pass && has_fail)
                .count();
            let flakiness_rate = if total_fps == 0 {
                0.0
            } else {
                flaky_fps as f64 / total_fps as f64
            };

            let median_duration_ms = median_u64(&mut acc.durations);
            snapshot.test_history.insert(
                test_id.clone(),
                TestHistory {
                    test_id,
                    total_runs: acc.total_runs,
                    total_failures: acc.total_failures,
                    ewma_fail_10: acc.ewma_10,
                    ewma_fail_50: acc.ewma_50,
                    runs_since_last_fail: acc.runs_since_last_fail,
                    median_duration_ms,
                    flakiness_rate,
                    last_run_ts: acc.last_run_ts,
                },
            );
        }

        // 2. Historical co-failure counts strictly prior to as_of_ts.
        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT o.test_id, d.file_path,
                           COUNT(*) AS co_runs,
                           SUM(CASE WHEN o.status IN ('fail', 'error') THEN 1 ELSE 0 END) AS co_fails
                    FROM test_outcomes o
                    JOIN runs r ON r.run_id = o.run_id
                    JOIN run_diff_files d ON d.run_id = o.run_id
                    WHERE r.started_at < ?1
                      AND o.status != 'skip'
                    GROUP BY o.test_id, d.file_path
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map(params![as_of_ts], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })
                .map_err(store_err)?;

            for row in rows {
                let (test_id, file_path, co_runs, co_fails) = row.map_err(store_err)?;
                snapshot.cofail_counts.insert(
                    (test_id, file_path),
                    CoFailCounts {
                        co_fails: co_fails.max(0) as u64,
                        co_runs: co_runs.max(0) as u64,
                    },
                );
            }
        }

        // 3. Git co-change frequencies & 30-day file churn strictly prior to as_of_ts.
        {
            let cutoff_30d = as_of_ts.saturating_sub(SECONDS_PER_30_DAYS);
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT cf.commit_sha, c.commit_ts, cf.file_path
                    FROM commit_files cf
                    JOIN commits c ON c.commit_sha = cf.commit_sha
                    WHERE c.commit_ts < ?1
                    ORDER BY cf.commit_sha ASC, cf.file_path ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map(params![as_of_ts], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })
                .map_err(store_err)?;

            let mut current_sha: Option<String> = None;
            let mut current_ts: i64 = 0;
            let mut current_files: Vec<String> = Vec::new();

            let flush_commit = |commit_ts: i64,
                                files: &[String],
                                file_counts: &mut BTreeMap<String, u32>,
                                pair_counts: &mut BTreeMap<(String, String), u32>,
                                churn_30d: &mut BTreeMap<String, u32>| {
                for f in files {
                    *file_counts.entry(f.clone()).or_insert(0) += 1;
                    if commit_ts >= cutoff_30d {
                        *churn_30d.entry(f.clone()).or_insert(0) += 1;
                    }
                }
                if files.len() <= MAX_COCHANGE_FILES_PER_COMMIT {
                    for i in 0..files.len() {
                        for j in (i + 1)..files.len() {
                            *pair_counts
                                .entry((files[i].clone(), files[j].clone()))
                                .or_insert(0) += 1;
                        }
                    }
                }
            };

            for row in rows {
                let (sha, commit_ts, file_path) = row.map_err(store_err)?;
                if current_sha.as_deref() != Some(sha.as_str()) {
                    if !current_files.is_empty() {
                        flush_commit(
                            current_ts,
                            &current_files,
                            &mut snapshot.file_commit_counts,
                            &mut snapshot.pair_commit_counts,
                            &mut snapshot.file_churn_30d,
                        );
                        current_files.clear();
                    }
                    current_sha = Some(sha);
                    current_ts = commit_ts;
                }
                current_files.push(file_path);
            }

            if !current_files.is_empty() {
                flush_commit(
                    current_ts,
                    &current_files,
                    &mut snapshot.file_commit_counts,
                    &mut snapshot.pair_commit_counts,
                    &mut snapshot.file_churn_30d,
                );
            }
        }

        Ok(snapshot)
    }

    /// Inserts a trained model metadata record into the `models` table.
    pub fn insert_model(&self, model: &ModelRecord) -> Result<(), PtsError> {
        self.conn
            .execute(
                "
                INSERT INTO models (
                    model_id, trained_at, max_train_ts, train_rows, pos_rows,
                    brier_score, calibration_a, calibration_b, artifact_path
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ",
                params![
                    model.model_id,
                    model.trained_at,
                    model.max_train_ts,
                    model.train_rows as i64,
                    model.pos_rows as i64,
                    model.brier_score,
                    model.calibration_a,
                    model.calibration_b,
                    model.artifact_path,
                ],
            )
            .map_err(store_err)?;
        Ok(())
    }

    /// Returns the most recently trained model metadata record, if any.
    pub fn get_latest_model(&self) -> Result<Option<ModelRecord>, PtsError> {
        self.conn
            .query_row(
                "
                SELECT model_id, trained_at, max_train_ts, train_rows, pos_rows,
                       brier_score, calibration_a, calibration_b, artifact_path
                FROM models
                ORDER BY trained_at DESC, rowid DESC
                LIMIT 1
                ",
                [],
                |r| {
                    Ok(ModelRecord {
                        model_id: r.get(0)?,
                        trained_at: r.get(1)?,
                        max_train_ts: r.get(2)?,
                        train_rows: r.get::<_, i64>(3)?.max(0) as usize,
                        pos_rows: r.get::<_, i64>(4)?.max(0) as usize,
                        brier_score: r.get(5)?,
                        calibration_a: r.get(6)?,
                        calibration_b: r.get(7)?,
                        artifact_path: r.get(8)?,
                    })
                },
            )
            .optional()
            .map_err(store_err)
    }

    /// Loads all historical runs in strict chronological order `(started_at ASC, run_id ASC)`
    /// along with their diff files and test outcomes.
    pub fn load_historical_runs(&self) -> Result<Vec<HistoricalRun>, PtsError> {
        let mut runs: Vec<HistoricalRun> = Vec::new();
        let mut run_index: BTreeMap<String, usize> = BTreeMap::new();

        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT run_id, commit_sha, diff_fingerprint, started_at
                    FROM runs
                    ORDER BY started_at ASC, run_id ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })
                .map_err(store_err)?;

            for row in rows {
                let (run_id, commit_sha, diff_fingerprint, started_at) = row.map_err(store_err)?;
                let idx = runs.len();
                run_index.insert(run_id.clone(), idx);
                runs.push(HistoricalRun {
                    run_id,
                    commit_sha,
                    diff_fingerprint,
                    started_at,
                    diff_files: Vec::new(),
                    outcomes: Vec::new(),
                });
            }
        }

        if runs.is_empty() {
            return Ok(runs);
        }

        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT run_id, file_path, lines_added, lines_deleted
                    FROM run_diff_files
                    ORDER BY run_id ASC, file_path ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })
                .map_err(store_err)?;

            for row in rows {
                let (run_id, path, lines_added, lines_deleted) = row.map_err(store_err)?;
                if let Some(&idx) = run_index.get(&run_id) {
                    runs[idx].diff_files.push(DiffFile {
                        path,
                        lines_added: lines_added.max(0) as u32,
                        lines_deleted: lines_deleted.max(0) as u32,
                    });
                }
            }
        }

        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT o.run_id, o.test_id, t.file_path, o.status, o.duration_ms, o.sampling_propensity
                    FROM test_outcomes o
                    JOIN tests t ON t.test_id = o.test_id
                    ORDER BY o.run_id ASC, o.test_id ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, f64>(5)?,
                    ))
                })
                .map_err(store_err)?;

            for row in rows {
                let (run_id, test_id, file_path, status_str, duration_ms, sampling_propensity) =
                    row.map_err(store_err)?;
                if let Some(&idx) = run_index.get(&run_id) {
                    runs[idx].outcomes.push(HistoricalOutcome {
                        test_id,
                        file_path,
                        status: TestStatus::from_str(&status_str)?,
                        duration_ms: duration_ms.max(0) as u64,
                        sampling_propensity,
                        ast_min_hop: None,
                    });
                }
            }
        }

        Ok(runs)
    }

    /// Loads all mined git commits ordered chronologically ascending by `(commit_ts, commit_sha)`.
    pub fn load_commits(&self) -> Result<Vec<CommitRecord>, PtsError> {
        let mut commits: Vec<CommitRecord> = Vec::new();
        let mut commit_index: BTreeMap<String, usize> = BTreeMap::new();

        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT commit_sha, parent_sha, author_hash, commit_ts,
                           files_changed, lines_added, lines_deleted
                    FROM commits
                    ORDER BY commit_ts ASC, commit_sha ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map([], |r| {
                    Ok(CommitRecord {
                        commit_sha: r.get(0)?,
                        parent_sha: r.get(1)?,
                        author_hash: r.get(2)?,
                        commit_ts: r.get(3)?,
                        files_changed: r.get::<_, i64>(4)?.max(0) as u32,
                        lines_added: r.get::<_, i64>(5)?.max(0) as u32,
                        lines_deleted: r.get::<_, i64>(6)?.max(0) as u32,
                        files: Vec::new(),
                    })
                })
                .map_err(store_err)?;

            for row in rows {
                let commit = row.map_err(store_err)?;
                let idx = commits.len();
                commit_index.insert(commit.commit_sha.clone(), idx);
                commits.push(commit);
            }
        }

        if commits.is_empty() {
            return Ok(commits);
        }

        {
            let mut stmt = self
                .conn
                .prepare(
                    "
                    SELECT commit_sha, file_path, lines_added, lines_deleted
                    FROM commit_files
                    ORDER BY commit_sha ASC, file_path ASC
                    ",
                )
                .map_err(store_err)?;

            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })
                .map_err(store_err)?;

            for row in rows {
                let (sha, path, lines_added, lines_deleted) = row.map_err(store_err)?;
                if let Some(&idx) = commit_index.get(&sha) {
                    commits[idx].files.push(DiffFile {
                        path,
                        lines_added: lines_added.max(0) as u32,
                        lines_deleted: lines_deleted.max(0) as u32,
                    });
                }
            }
        }

        Ok(commits)
    }
}

fn median_u64(values: &mut [u64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid] as f64)
    } else {
        Some((values[mid - 1] as f64 + values[mid] as f64) / 2.0)
    }
}

type RawOutcomeRow = (
    String,
    String,
    String,
    i64,
    Option<f64>,
    String,
    f64,
    Option<String>,
);

fn map_outcome_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<RawOutcomeRow> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
        r.get(6)?,
        r.get(7)?,
    ))
}

fn collect_outcomes<I>(rows: I) -> Result<Vec<TestOutcomeRecord>, PtsError>
where
    I: Iterator<Item = rusqlite::Result<RawOutcomeRow>>,
{
    let mut outcomes = Vec::new();
    for row in rows {
        let (
            run_id,
            test_id,
            status_str,
            duration_ms,
            predicted_prob,
            reason_str,
            sampling_propensity,
            failure_excerpt,
        ) = row.map_err(store_err)?;
        outcomes.push(TestOutcomeRecord {
            run_id,
            test_id,
            status: TestStatus::from_str(&status_str)?,
            duration_ms: duration_ms.max(0) as u64,
            predicted_prob,
            selection_reason: SelectionReason::from_str(&reason_str)?,
            sampling_propensity,
            failure_excerpt,
        });
    }
    Ok(outcomes)
}

fn store_err(e: rusqlite::Error) -> PtsError {
    PtsError::Store(e.to_string())
}

fn current_unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
