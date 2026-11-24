# PTS — Predictive Test Selector

**PTS (`pts`)** is a single-binary, pure-Rust CLI and local stdio MCP server that predicts which subset of a repository's test suite is most likely to fail for a given working-tree `git diff`, runs the highest-value tests within a strict time or count budget, and records outcomes locally to improve over time.

- **Zero external infrastructure:** Single self-contained Rust binary (Edition 2024). Statically linked SQLite (`rusqlite` bundled), pure-Rust GBDT (`forust-ml`) + Platt calibration, and Tree-sitter Python AST import analysis—no Python ML dependencies, no database servers, no cloud API calls.
- **Cost-aware selection with anti-starvation:** Maximizes failure probability per millisecond (`prob / eff_ms`) with a deterministic staleness-weighted exploration pass so cold tests never starve and inverse-propensity weighting keeps training unbiased.
- **Leakage-free learning & evaluation:** Strict chronological run-boundary splits (`max_train_ts < min_cal_ts < min_eval_ts`) across both training and walk-forward evaluation.
- **Built for both humans and coding agents:** Identical execution service powers CLI subcommands (`pts run`, `pts explain`) and the local `rmcp` stdio MCP server (`pts mcp`).

---

## Architecture

```text
                    +-------------------------------------------+
                    |             pts (pts-cli)                 |
                    |   Clap CLI    |  service.rs  |  rmcp MCP  |
                    +---------------+------+-------+------------+
                                           |
          +--------------------------------+--------------------------------+
          |                                |                                |
          v                                v                                v
+--------------------+         +-----------------------+         +---------------------+
|   pts-collectors   |         |        pts-ml         |         |     pts-runners     |
| - git diff/history |         | - 16 pure features    |         | - PytestRunner      |
| - tree-sitter Py   |         | - Cold-start ranker   |         | - JUnit XML parser  |
|   AST import graph |         | - forust-ml GBDT      |         | - Bounded tracebacks|
| - TempGitWorktree  |         | - Platt calibration   |         +---------------------+
+---------+----------+         | - Cost-aware selector |                    |
          |                    +-----------+-----------+                    |
          +----------------------+         |         +----------------------+
                                 v         v         v
                    +--------------------+   +--------------------+
                    |     pts-store      |   |      pts-eval      |
                    | - Bundled SQLite   |   | - Walk-forward     |
                    | - WAL + migrations |   | - Random/AST/EWMA  |
                    | - As-of snapshots  |   |   baselines        |
                    +----------+---------+   +----------+---------+
                               |                        |
                               +-----------+------------+
                                           v
                              +-------------------------+
                              |        pts-core         |
                              | Pure types, Config,     |
                              | TestRunner trait, Error |
                              +-------------------------+
```

### Crate Overview

| Crate | Responsibility |
| :--- | :--- |
| `pts-core` | Pure domain contracts (`TestCase`, `DiffSummary`, `HistorySnapshot`, `Config`, `TestRunner`, `PtsError`). Zero I/O or subprocesses. |
| `pts-store` | Bundled SQLite store (WAL mode), schema migrations, atomic run/batch persistence, and strictly point-in-time (`< as_of_ts`) history queries. |
| `pts-collectors` | Working-tree `git diff` / `git log` miner, isolated `TempGitWorktree` for safe backfilling, and Tree-sitter Python file-level import graph builder. |
| `pts-ml` | Pure 16-feature extractor, cold-start heuristic ranker, `forust-ml` GBDT trainer, Newton-Raphson Platt calibrator, and deterministic cost-aware selector. |
| `pts-runners` | `TestRunner` implementation for `pytest` with JUnit XML parsing and bounded failure tracebacks. |
| `pts-eval` | Zero-leakage chronological walk-forward evaluator comparing PTS against `Random`, `StaticAst`, and `RecentFailureRate` baselines. |
| `pts-cli` | The `pts` binary: shared `service.rs`, Clap CLI subcommands, JSONL CI importer, and `rmcp` stdio MCP server. |

---

## Installation

Requires Rust 1.88+ (Edition 2024) and `pytest` available on `PATH` (or configured via `.pts.toml`).

```bash
cargo install --path crates/pts-cli
```

Verify installation:

```bash
pts --help
```

---

## CLI Quickstart

### 1. Initialize a repository

Creates `.pts/store.db` and mines recent git commit history for co-change and 30-day churn features:

```bash
pts init --history-limit 500
```

> Add `.pts/` to your repository's `.gitignore`.

### 2. Run tests with predictive selection

```bash
# Run the full suite once to seed timing and baseline outcomes
pts run --all

# Select and run highest-value tests within a 15-second budget
pts run --budget 15

# Preview selection without executing tests or writing to SQLite
pts run --budget 15 --dry-run --json

# Alternative selection modes
pts run --top-k 20
pts run --target-recall 0.95
```

When executed tests fail, `pts run` always parses the JUnit XML and commits all outcomes to `.pts/store.db` before exiting with `pytest`'s non-zero status code.

### 3. Train the GBDT + Platt model

Once history is collected (automatic activation threshold defaults to `min_runs_for_ml = 100` and `min_failures_for_ml = 25`, configurable via `.pts.toml`), train the model:

```bash
pts train
```

This writes `.pts/model.json` and records calibration & Brier score metadata in SQLite.

### 4. Inspect why a test was ranked (`pts explain`)

Inspect all 16 extracted features, active model mode (`gbdt` or `heuristic`), raw GBDT margin, calibrated failure probability, yield score, and top contributing signals:

```bash
pts explain tests/test_calc.py::test_add_positive
pts explain tests/test_calc.py::test_add_positive --json
```

### 5. Populate history from git commits or CI/PR logs

**Option A: Replay recent git commits in an isolated worktree** (never touches your working tree):

```bash
pts backfill --commits 50 --ref HEAD
```

**Option B: Import external CI / PR failure logs from JSONL** (no git checkout required):

```bash
pts import-runs ci_history.jsonl
```

Each line of the JSONL file represents one historical test run and is ingested atomically into `.pts/store.db` with `mode = 'backfill'`:

```json
{"commit_sha":"a1b2c3d","started_at":1710000000,"diff_files":[{"path":"src/calc.py","lines_added":18,"lines_deleted":2}],"outcomes":[{"test_id":"tests/test_calc.py::test_divide_broken","file_path":"tests/test_calc.py","status":"fail","duration_ms":130}]}
```

### 6. Run chronological walk-forward evaluation (`pts eval`)

Compare PTS against `Random`, `StaticAst`, and `RecentFailureRate` baselines across multiple time budgets with strict zero-leakage verification:

```bash
pts eval --budgets 0.1,0.2,0.3 --min-train-runs 20 --retrain-every 5 --output-json eval_report.json
```

---

## Configuration (`.pts.toml`)

All fields are optional; copy `.pts.toml.example` to `.pts.toml` at your repository root to override defaults:

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

## MCP Server (`pts mcp`)

`pts mcp` runs a local [Model Context Protocol](https://modelcontextprotocol.io) server over `stdio` using `rmcp`. `stdout` is strictly reserved for JSON-RPC 2.0 framing; all tracing logs and subprocess diagnostics are isolated from `stdout`.

### Exposed Tools

| Tool | Description |
| :--- | :--- |
| `select_tests` | Dry-run test selection for the current working-tree diff (`budget_seconds`, `top_k`, `target_recall`). |
| `run_fast_tests` | Select and execute tests within a budget or top-k limit, persist outcomes to `.pts/store.db`, and return structured pass/fail counts with bounded tracebacks (`budget_seconds`, `top_k`, `run_all`). |
| `explain_selection` | Return all 16 features, active model mode, calibrated failure probability, yield score, and top contributing signals for a specific `test_id`. |

### Client Configuration (Stdio)

Add PTS to your MCP client configuration (Claude Desktop, Cursor, Claude Code, Antigravity, etc.), pointing `cwd` at your target repository:

```json
{
  "mcpServers": {
    "pts": {
      "command": "pts",
      "args": ["mcp"],
      "cwd": "/path/to/your/python-repo"
    }
  }
}
```

---

## Benchmarks

> **Status:** Ready for real OSS Python repository evaluation results.

Reproduce a walk-forward benchmark on any Python repository with pre-merge CI/PR logs (via `pts import-runs`) or commit replay (via `pts backfill`):

```bash
pts init --history-limit 1000
pts import-runs path/to/oss_pr_runs.jsonl
pts eval --budgets 0.1,0.2,0.3 --min-train-runs 50 --retrain-every 10 --output-json benchmark.json
```

### OSS Evaluation Results (Template)

| Repository | Evaluated Runs | Budget | Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| _TBD (e.g. `httpx` / `pytest`)_ | — | 10% | **PTS (GBDT+Platt)** | — | — | — | — | — |
| _TBD_ | — | 10% | StaticAst | — | — | — | — | — |
| _TBD_ | — | 10% | RecentFailureRate | — | — | — | — | — |
| _TBD_ | — | 10% | Random | — | — | — | — | — |

---

## Development

Run the full workspace test suite and linter:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
