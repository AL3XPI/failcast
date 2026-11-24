use std::path::Path;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use pts_core::{DiffFile, PtsError, RunMode, SelectionReason, TestCase, TestStatus};
use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA_SQL: &str = include_str!("schema.sql");
const SCHEMA_VERSION: i64 = 1;

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
