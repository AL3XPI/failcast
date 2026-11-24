# PTS (Predictive Test Selector) — v1 Technical Specification

## 1. Objective & Product Boundaries

**PTS (`pts`)** is a single-binary, pure-Rust CLI and local stdio MCP server that predicts which subset of a repository's test suite is most likely to fail for a given working-tree `git diff`, runs the optimal subset within a strict time or count budget, and records outcomes locally to improve over time.

### Non-Negotiable Engineering Invariants
1. **Single Self-Contained Binary:** Pure Rust (Edition 2024). Zero Python ML dependencies, zero external database servers, zero cloud API calls, zero GPU requirements. SQLite is statically linked via `rusqlite` (`features = ["bundled"]`).
2. **Strict Decoupling:** `pts-ml` and `pts-core` MUST remain 100% pure—no subprocesses, no filesystem I/O, no network I/O, no wall-clock `SystemTime::now()` calls inside feature extraction or ranking logic (pass timestamps explicitly).
3. **Determinism:** Identical `(diff, store_snapshot, model, config)` inputs MUST produce identical test orderings and selections. Fix all RNG seeds (`seed = 42` for training; deterministic hash of `diff_fingerprint` for exploration) and break all score ties lexicographically by `test_id`.
4. **Zero Data Leakage:** Training, calibration, and walk-forward evaluation MUST partition strictly by monotonic timestamp (`max_train_ts < min_cal_ts < min_eval_ts`). Random k-fold splitting across time is forbidden.
5. **Stdio MCP Purity:** When running `pts mcp`, `stdout` is exclusively owned by JSON-RPC framing. All `tracing` logs, warnings, and runner diagnostics MUST route strictly to `stderr`.
6. **Persist Before Exit:** When executed tests fail (`pytest` exit code 1), `pts run` MUST parse the JUnit XML, commit all pass/fail outcomes to `.pts/store.db`, emit its report, and only then exit with the runner's non-zero status code.

### v1 Scope Constraints
* **Supported test runner:** `pytest` only (abstracted behind the `TestRunner` trait so `cargo`, `jest`, and `go` slot in cleanly in v2).
* **Supported AST language:** Python only via `tree-sitter-python` (abstracted behind an `ImportGraphBuilder` trait).
* **State directory:** `.pts/` inside repo root (`store.db`, `model.json`, `config.toml` optional override in repo root as `.pts.toml`).

---

## 2. Cargo Workspace Layout

```text
pts/
├── Cargo.toml                     # Workspace root (resolver = "3", edition = "2024")
├── SPEC.md
├── .pts.toml.example
├── crates/
│   ├── pts-core/                  # Pure domain types, Config, TestRunner trait, errors
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs
│   │       ├── config.rs
│   │       ├── runner.rs
│   │       └── error.rs
│   ├── pts-store/                 # SQLite migrations, WAL setup, typed repository queries
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── schema.sql
│   │       └── store.rs
│   ├── pts-collectors/            # Git diff/history miner + Tree-sitter Python AST graph
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── git.rs
│   │       └── ast_python.rs
│   ├── pts-ml/                    # Pure feature engineering, heuristic, forust-ml, Platt, selector
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── features.rs
│   │       ├── heuristic.rs
│   │       ├── gbdt.rs
│   │       ├── platt.rs
│   │       └── selector.rs
│   ├── pts-runners/               # PytestRunner implementation + JUnit XML parser
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── pytest.rs
│   │       └── junit.rs
│   ├── pts-eval/                  # Chronological walk-forward evaluator & baselines
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── baselines.rs
│   │       ├── metrics.rs
│   │       └── walk_forward.rs
│   └── pts-cli/                   # Binary entrypoint (`pts`): Clap CLI + rmcp stdio server
│       └── src/
│           ├── main.rs
│           ├── cli.rs
│           └── mcp.rs
└── tests/
    └── fixtures/
        └── mini_py_repo/          # Deterministic fixture Python project for integration tests
```

### Core Crate Dependencies (Workspace Root)
* CLI & Async: `clap` (derive), `tokio` (`rt-multi-thread`, `macros`, `io-std`, `process`), `tracing`, `tracing-subscriber` (writing to `stderr`).
* Serialization & Errors: `serde` (derive), `serde_json`, `toml`, `thiserror`, `anyhow` (CLI binary only), `schemars` (JSON Schema for MCP).
* Storage: `rusqlite` (`bundled`), `uuid` (`v4`).
* Parsing & Collectors: `tree-sitter`, `tree-sitter-python`, `quick-xml` (JUnit XML), `sha2` (diff fingerprinting).
* ML: `forust-ml` (pure-Rust GBDT).
* MCP: `rmcp` (official Rust MCP SDK over stdio).
* Dev/Test: `tempfile`, `assert_cmd`, `predicates`.

> **Agent Crate-Version Note:** Before writing code that calls `forust-ml` or `rmcp`, inspect the resolved versions in `Cargo.lock` / docs so exact builder/macro signatures match the installed versions while keeping all behavioral contracts in this spec unchanged.

---

## 3. Core Domain Contracts (`pts-core`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    Pass,
    Fail,
    Error,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionReason {
    Full,
    Heuristic,
    Ranked,
    Explore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffFile {
    pub path: String,          // Repo-relative POSIX path ("src/auth/token.py")
    pub lines_added: u32,
    pub lines_deleted: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub base_commit: Option<String>,
    pub diff_fingerprint: String, // SHA-256 of canonical sorted diff paths + line counts
    pub files: Vec<DiffFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub test_id: String,       // Canonical runner ID, e.g. "tests/test_auth.py::test_expired"
    pub file_path: String,     // Repo-relative POSIX path ("tests/test_auth.py")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionOutcome {
    pub test_id: String,
    pub status: TestStatus,
    pub duration_ms: u64,
    pub failure_message: Option<String>, // Truncated to 1,500 chars max
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerResult {
    pub exit_code: i32,
    pub total_duration_ms: u64,
    pub outcomes: Vec<TestExecutionOutcome>,
}

pub trait TestRunner: Send + Sync {
    fn name(&self) -> &'static str;
    fn discover(&self, repo_root: &std::path::Path) -> Result<Vec<TestCase>, PtsError>;
    fn execute(
        &self,
        repo_root: &std::path::Path,
        selected_ids: Option<&[String]>, // None = run entire suite
    ) -> Result<RunnerResult, PtsError>;
}
```

### Configuration (`.pts.toml`)
All fields optional with safe defaults:
```toml
runner = "pytest"
default_budget_secs = 15
explore_fraction = 0.10
min_failures_for_ml = 25
min_runs_for_ml = 100
history_commit_limit = 500
pytest_bin = "pytest"
max_traceback_chars = 1500
```

---

## 4. SQLite Schema (`crates/pts-store/src/schema.sql`)

Connections MUST execute these pragmas on open:
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
```

```sql
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
```

---

## 5. ML Architecture: Features, `forust-ml`, Platt Calibration & Selector

### 5.1 Feature Vector (`NUM_FEATURES = 16`, fixed `f64` order)
Missing or undefined values MUST be encoded as `f64::NAN` (natively routed by `forust-ml` decision trees).

| Index | Name | Definition |
| :--- | :--- | :--- |
| 0 | `ast_min_hop` | Shortest directed import-graph hop distance from `test.file_path` to any modified file (`NAN` if unreachable; `0.0` if test file itself modified) |
| 1 | `direct_test_modified` | `1.0` if `test.file_path` is in `diff.files`, else `0.0` |
| 2 | `path_jaccard_max` | Max token Jaccard similarity between `test.file_path` path/stem tokens and any modified file |
| 3 | `same_dir_overlap` | `1.0` if any modified file shares parent directory with `test.file_path`, else `0.0` |
| 4 | `cofail_rate_max` | Max historical $P(\text{test fails} \mid \text{file } f \text{ in diff})$ across modified files (Laplace-smoothed: $(\text{cofails} + 0.5) / (\text{co\_runs} + 5.0)$) |
| 5 | `cochange_freq_max` | Max git commit co-occurrence ratio between `test.file_path` and any modified file over history |
| 6 | `ewma_fail_10` | Exponentially weighted failure rate of `test_id` ($\alpha = 0.2$, fast decay) |
| 7 | `ewma_fail_50` | Exponentially weighted failure rate of `test_id` ($\alpha = 0.04$, medium decay) |
| 8 | `runs_since_last_fail` | Count of passing runs since last failure of `test_id` (`NAN` if never failed) |
| 9 | `total_test_runs` | Total historical runs observed for `test_id` |
| 10 | `median_duration_ms` | Historical median duration of `test_id` in ms (`NAN` if unseen) |
| 11 | `flakiness_rate` | Fraction of commits where `test_id` produced both `pass` and `fail` on identical `diff_fingerprint` |
| 12 | `diff_files_count` | Number of modified files in diff |
| 13 | `diff_lines_total` | `log1p(sum(lines_added + lines_deleted))` |
| 14 | `diff_top_dirs_count` | Distinct top-level directories touched by diff |
| 15 | `touched_file_churn_30d` | Max 30-day commit count among modified files |

### 5.2 Cold-Start Heuristic Ranker
Used automatically when `total_runs < config.min_runs_for_ml` OR `total_failures < config.min_failures_for_ml` OR no trained model exists:
```rust
// Deterministic bounded score in (0.0, 1.0)
let raw = 3.0 * direct_test_modified
    + 2.5 * ast_hop_decay(ast_min_hop)     // 1.0 / (1.0 + hop), 0.0 if NAN
    + 1.5 * path_jaccard_max
    + 1.5 * cochange_freq_max
    + 1.0 * ewma_fail_10;
let prob = 1.0 / (1.0 + (- (raw - 1.5)).exp());
```

### 5.3 `forust-ml` GBDT Training Contract
* **CRITICAL MATRIX LAYOUT:** `forust_ml::Matrix::new(&data, rows, cols)` expects **column-major** flat storage:
  `data[col * rows + row] = feature_matrix[row][col]`.
  Write a dedicated `to_column_major(rows: &[[f64; 16]]) -> Vec<f64>` helper with a unit test verifying layout.
* **Chronological Split:**
  1. Sort all historical `(run, test_outcome)` rows strictly ascending by `runs.started_at`, tie-breaking by `(run_id, test_id)`.
  2. Partition at the **run boundary** (never split tests from the same run across train and calibration): earliest 80% of runs go to `train_set`, latest 20% of runs go to `cal_set`.
  3. Binary label: `y = 1.0` if `status` in `{'fail', 'error'}`, `y = 0.0` if `status == 'pass'` (`skip` rows are excluded from training).
  4. Inverse-propensity sample weighting: `w_i = (1.0 / sampling_propensity).clamp(1.0, 10.0)` to counter selection bias from previous partial runs.
* **Hyperparameters (v1 defaults):**
  * `objective_type`: `LogLoss`
  * `iterations`: `150`
  * `learning_rate`: `0.05`
  * `max_depth`: `5`
  * `min_leaf_weight`: `5.0`
  * `l2`: `2.0`
  * `seed`: `42` (or deterministic single-threaded / fixed-seed build)
* **Raw Margin Check:** Add a unit test asserting whether `GradientBooster::predict` outputs raw log-odds margins (which can be negative or > 1.0) vs. sigmoid probabilities. Platt scaling takes raw log-odds `z_i`; if the installed version returns probabilities `p`, convert via clamped logit `z = ln(p / (1 - p))`.

### 5.4 Platt Scaling Calibrator (`crates/pts-ml/src/platt.rs`)
Fit $P(y=1 \mid z) = \sigma(a z + b) = \frac{1}{1 + e^{-(a z + b)}}$ on the **unweighted** chronological `cal_set` so predicted probabilities reflect real-world base rates.

* **Target smoothing (Platt 1999):** Let $N_+$ be positive count and $N_-$ be negative count in `cal_set`:
  $$t_+ = \frac{N_+ + 1}{N_+ + 2}, \qquad t_- = \frac{1}{N_- + 2}$$
* **Objective:** Minimize L2-regularized negative log-likelihood ($\lambda = 10^{-4}$):
  $$\mathcal{L}(a, b) = -\sum_{i=1}^{n} \left[ t_i \ln p_i + (1 - t_i) \ln(1 - p_i) \right] + \frac{\lambda}{2}(a^2 + b^2)$$
* **Optimizer:** Damped Newton-Raphson solving the closed-form 2x2 Hessian system (`max_iter = 50`, `tol = 1e-7`, backtracking line search).
* **Monotonicity Guardrail:** Enforce `a >= 0.0` (if optimization yields `a < 0.0`, fall back to `a = 1.0, b = logit(base_rate)`).
* **Serialization:** Persist `ModelArtifact { version, trained_at, max_train_ts, feature_names, booster_json, platt_a, platt_b }` to `.pts/model.json`.

### 5.5 Cost-Aware Selector & Anti-Starvation (`crates/pts-ml/src/selector.rs`)
Given candidate tests with `(test_id, prob, est_duration_ms, days_since_last_run)` and `budget_ms`:
1. **Duration Floor:** `eff_ms = est_duration_ms.unwrap_or(median_repo_test_ms).max(50.0)`.
2. **Budget Split:**
   * `explore_budget_ms = (budget_ms as f64 * config.explore_fraction) as u64`
   * `exploit_budget_ms = budget_ms.saturating_sub(explore_budget_ms)`
3. **Exploit Pass:** Sort candidates by `yield = prob / eff_ms` descending, tie-break by `test_id` ascending. Greedily add tests while `spent_ms + eff_ms <= exploit_budget_ms` (always select at least the single top-ranked test if candidate list is non-empty). Mark `SelectionReason::Ranked` (or `Heuristic` in cold-start) with `sampling_propensity = 1.0`.
4. **Explore Pass:** From remaining unselected tests, assign staleness weight $w_i = 1.0 + \text{days\_since\_last\_run}_i$. Using a deterministic SplitMix64 PRNG seeded from the first 8 bytes of `diff_fingerprint`, sample tests without replacement proportional to $w_i$ while `explore_spent_ms + eff_ms <= explore_budget_ms`. Record `SelectionReason::Explore` and `sampling_propensity = (w_i / sum_w).clamp(0.01, 1.0)`.
5. **Target-Recall Mode (`--target-recall R`):** Alternatively, accumulate tests in descending `prob` order until $1 - \prod_{t \in \text{unselected}} (1 - \hat{p}_t) \le (1 - R)$, or until `max_budget_ms` cap is hit.

---

## 6. CLI & `rmcp` Stdio Tool Contracts

### 6.1 CLI Subcommands (`pts-cli`)
* `pts init [--history-limit 500]` — Creates `.pts/`, initializes SQLite, mines recent git commit history + co-change index, ensures `.pts/` is noted in `.gitignore` warning if untracked.
* `pts run (--all | --budget <SECS> | --top-k <N> | --target-recall <0..1>) [--dry-run] [--json]` — Computes diff, selects tests, invokes runner, persists outcomes, prints summary table (or JSON).
* `pts train` — Trains `forust-ml` + Platt calibrator from `.pts/store.db`, writes `.pts/model.json`, records metrics in `models` table.
* `pts explain <TEST_ID>` — Prints feature breakdown and score rationale for `<TEST_ID>` against current diff.
* `pts eval --output-json <PATH>` — Runs chronological walk-forward benchmark comparing PTS against `Random`, `StaticAst`, and `RecentFailureRate` baselines; prints Markdown comparison table.
* `pts mcp` — Starts MCP server over stdio.

### 6.2 `rmcp` Tool Contracts (`crates/pts-cli/src/mcp.rs`)
All schemas derived via `#[derive(Deserialize, Serialize, schemars::JsonSchema)]`.

#### Tool 1: `select_tests`
Dry-run test selection for the current working-tree diff (fast, sub-50ms, does not execute tests).
* **Input:**
  ```json
  {
    "budget_seconds": 15,
    "top_k": null,
    "target_recall": null
  }
  ```
* **Output:**
  ```json
  {
    "model_mode": "gbdt | heuristic",
    "diff_files_count": 3,
    "total_discovered_tests": 412,
    "selected_count": 24,
    "estimated_duration_ms": 13400,
    "selected": [
      {
        "test_id": "tests/test_auth.py::test_refresh_token",
        "predicted_fail_prob": 0.412,
        "est_duration_ms": 180,
        "reason": "ranked"
      }
    ]
  }
  ```

#### Tool 2: `run_fast_tests`
Selects and executes the highest-value tests within the time budget, saves results to `.pts/store.db`, and returns structured outcomes with bounded tracebacks.
* **Input:**
  ```json
  {
    "budget_seconds": 15,
    "top_k": null,
    "run_all": false
  }
  ```
* **Output:**
  ```json
  {
    "run_id": "uuid-v4",
    "model_mode": "gbdt | heuristic",
    "passed": 22,
    "failed": 2,
    "errored": 0,
    "skipped": 0,
    "total_duration_ms": 11840,
    "failures": [
      {
        "test_id": "tests/test_auth.py::test_refresh_token",
        "duration_ms": 195,
        "failure_message": "AssertionError: expected 401, got 200 ..."
      }
    ]
  }
  ```

#### Tool 3: `explain_selection`
Inspects the exact features and score behind a test's ranking for the current diff.
* **Input:**
  ```json
  {
    "test_id": "tests/test_auth.py::test_refresh_token"
  }
  ```
* **Output:**
  ```json
  {
    "test_id": "tests/test_auth.py::test_refresh_token",
    "model_mode": "gbdt | heuristic",
    "predicted_fail_prob": 0.412,
    "est_duration_ms": 180,
    "yield_score": 0.00229,
    "features": {
      "ast_min_hop": 1.0,
      "direct_test_modified": 0.0,
      "path_jaccard_max": 0.5,
      "cofail_rate_max": 0.35,
      "ewma_fail_10": 0.22
    }
  }
  ```

---

## 7. Milestones & Milestone 1 Acceptance Tests

### Milestone Roadmap
* **M1 (Foundation):** Workspace scaffold, `pts-core`, `pts-store`, `pts-runners` (`pytest` + JUnit XML), `pts init` (DB setup only), `pts run --all`.
* **M2 (Collectors & Cold-Start):** `pts-collectors` (git diff/history + `tree-sitter-python` import graph), pure `features.rs`, `heuristic.rs`, `selector.rs`, `pts run --budget` and `--dry-run`.
* **M3 (ML & Walk-Forward Eval):** `forust-ml` trainer, Platt calibrator, `pts train`, `pts eval` vs. baselines, strict anti-leakage tests.
* **M4 (MCP & Polish):** `rmcp` stdio server (`pts mcp`), `pts explain`, `README.md` with reproducible benchmark output on a real OSS Python repo.

### Milestone 1 Acceptance Tests (Gate Before Starting M2)
Every test below MUST exist and pass via `cargo test --workspace` before Milestone 1 is declared complete.

#### Vendored Fixture Repo (`tests/fixtures/mini_py_repo/`)
Contains:
* `src/calc.py` (`add`, `divide`)
* `tests/test_calc.py` with 4 tests:
  1. `test_add_positive` (passes)
  2. `test_add_negative` (passes)
  3. `test_divide_broken` (fails when env var `PTS_FIXTURE_FAIL=1`, passes otherwise—lets tests deterministically trigger both green and red runs)
  4. `test_skipped_feature` (`@pytest.mark.skip(reason="v2")`)

#### Required Tests
1. **`pts_store::tests::schema_init_is_idempotent_and_enables_wal`**
   * Opens a temp SQLite DB via `Store::open()`, calls `migrate()` twice.
   * Asserts `PRAGMA journal_mode` returns `"wal"` and `PRAGMA foreign_keys` returns `1`.
   * Asserts all 8 tables exist and `schema_migrations` has exactly 1 row (`version = 1`).
2. **`pts_store::tests::roundtrip_run_and_outcomes`**
   * Inserts a run, 2 diff files, 3 tests, and 3 outcomes (`pass`, `fail`, `skip`).
   * Queries back outcomes by `run_id` and by `test_id`, asserting exact equality on `status`, `duration_ms`, `selection_reason`, and `sampling_propensity`.
   * Asserts invalid status string violates the `CHECK` constraint and returns an error.
3. **`pts_runners::tests::junit_xml_parses_pass_fail_error_skip_and_truncates`**
   * Pure unit test (no `pytest` needed) feeding a multi-case JUnit XML fixture containing `<failure>`, `<error>`, `<skipped>`, and a 5,000-char traceback.
   * Asserts canonical `test_id` mapping, accurate `duration_ms` conversion (`seconds * 1000`), and traceback truncation to `<= max_traceback_chars`.
4. **`pts_runners::tests::pytest_runner_discover_and_execute_fixture`**
   * Runs `PytestRunner::discover` on `tests/fixtures/mini_py_repo`, asserting all 4 node IDs are discovered and normalized to POSIX paths.
   * Runs `PytestRunner::execute` with `PTS_FIXTURE_FAIL=1`: asserts `exit_code == 1`, 2 passes, 1 fail (non-empty `failure_message`), 1 skip.
   * Runs `PytestRunner::execute` with `selected_ids = Some(&["tests/test_calc.py::test_add_positive"])`: asserts `exit_code == 0` and exactly 1 outcome.
5. **`pts_cli::tests::cli_run_all_persists_even_on_test_failure`**
   * Copies `mini_py_repo` to a `TempDir`, runs `pts init`, then runs `pts run --all` with `PTS_FIXTURE_FAIL=1` using `assert_cmd`.
   * Asserts CLI exits with code `1` (propagating test failure).
   * Opens `.pts/store.db` directly and asserts `runs` has 1 row (`exit_code = 1`, `mode = 'all'`) and `test_outcomes` has all 4 rows persisted cleanly.
6. **`pts_core::tests::config_defaults_and_toml_override`**
   * Verifies `Config::load` returns defaults when `.pts.toml` is absent and overrides only specified keys when partial TOML is present.