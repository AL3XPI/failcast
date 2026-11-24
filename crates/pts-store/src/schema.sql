CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS commits (
    commit_sha TEXT PRIMARY KEY,
    parent_sha TEXT,
    author_hash TEXT NOT NULL,
    commit_ts INTEGER NOT NULL,
    files_changed INTEGER NOT NULL,
    lines_added INTEGER NOT NULL,
    lines_deleted INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS commit_files (
    commit_sha TEXT NOT NULL REFERENCES commits(commit_sha) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    lines_added INTEGER NOT NULL,
    lines_deleted INTEGER NOT NULL,
    PRIMARY KEY (commit_sha, file_path)
);

CREATE TABLE IF NOT EXISTS tests (
    test_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    first_seen_ts INTEGER NOT NULL,
    last_seen_ts INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    commit_sha TEXT,
    diff_fingerprint TEXT NOT NULL,
    runner TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('all', 'budget', 'top_k', 'backfill')),
    budget_ms INTEGER,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    total_duration_ms INTEGER,
    exit_code INTEGER
);

CREATE TABLE IF NOT EXISTS run_diff_files (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    lines_added INTEGER NOT NULL,
    lines_deleted INTEGER NOT NULL,
    PRIMARY KEY (run_id, file_path)
);

CREATE TABLE IF NOT EXISTS test_outcomes (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_id TEXT NOT NULL REFERENCES tests(test_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('pass', 'fail', 'error', 'skip')),
    duration_ms INTEGER NOT NULL,
    predicted_prob REAL,
    selection_reason TEXT NOT NULL CHECK(selection_reason IN ('full', 'heuristic', 'ranked', 'explore')),
    sampling_propensity REAL NOT NULL DEFAULT 1.0,
    failure_excerpt TEXT,
    PRIMARY KEY (run_id, test_id)
);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    trained_at INTEGER NOT NULL,
    max_train_ts INTEGER NOT NULL,
    train_rows INTEGER NOT NULL,
    pos_rows INTEGER NOT NULL,
    brier_score REAL NOT NULL,
    calibration_a REAL NOT NULL,
    calibration_b REAL NOT NULL,
    artifact_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_commits_ts ON commits(commit_ts);
CREATE INDEX IF NOT EXISTS idx_commit_files_path ON commit_files(file_path);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_test_run ON test_outcomes(test_id, run_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_status ON test_outcomes(status);
