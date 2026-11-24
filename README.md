# PTS — Predictive Test Selector

[![CI](https://github.com/AL3XPI/pts/actions/workflows/ci.yml/badge.svg)](https://github.com/AL3XPI/pts/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

### Empirical Walk-Forward Evaluation (`PyGithub/PyGithub`)

- **Dataset transparency:** Collected from real pre-merge Pull Request runs of [`PyGithub/PyGithub`](https://github.com/PyGithub/PyGithub) (`CI` workflow) spanning **2026-09-09 to 2026-11-20** using [`scripts/collect_gha_pytest.py`](scripts/collect_gha_pytest.py).
  - **80 PR runs** total (**22 warmup** + **58 strictly chronological evaluated runs**; zero timestamp leakage verified on every step).
  - **19 failing PR runs** total (**13 failing runs** / **335 individual test failures** in the evaluation window).
  - **1,405 unique `pytest` tests** across **93,736 total outcomes** (**68,695 evaluated non-skip outcomes**).
  - **100.0% exact per-test durations** sourced directly from GitHub Actions JUnit XML (`pytest.xml`) workflow artifacts (`0.0%` estimated; 2 zero-pass environment/collection-crash runs excluded).
- **Model cadence:** GBDT (`forust-ml`, 150 trees, depth 5) + chronological 80/20 Platt calibration retrained every 5 runs (`gbdt_steps = 58 / 58`). Full structured report saved in [`docs/benchmarks/eval_report.json`](docs/benchmarks/eval_report.json).

![Failure Detection Rate vs. Time Budget — PyGithub/PyGithub](docs/benchmarks/pareto_fdr_vs_budget.png)

#### Chronological Walk-Forward Comparison (`PyGithub/PyGithub`, 58 Evaluated Runs, 13 Failing Runs)

| Budget | Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **5%** | **PTS (GBDT+Platt)** | **61.5%** | **26.3%** | **98.1%** | **0.0047** | **0.4850** |
| 5% | Heuristic | 46.2% | 5.7% | 98.7% | 0.2035 | 0.0115 |
| 5% | StaticAst | 38.5% | 6.3% | 98.7% | 0.1424 | 0.0059 |
| 5% | RecentFailureRate | 69.2% | 17.9% | 97.1% | 0.0042 | 0.3085 |
| 5% | Random | 30.8% | 3.3% | 97.9% | 0.3304 | 0.0051 |
| **10%** | **PTS (GBDT+Platt)** | **69.2%** | **50.1%** | **96.0%** | **0.0047** | **0.4850** |
| 10% | Heuristic | 46.2% | 11.9% | 96.8% | 0.2035 | 0.0115 |
| 10% | StaticAst | 38.5% | 9.0% | 97.0% | 0.1424 | 0.0059 |
| 10% | RecentFailureRate | 69.2% | 30.1% | 94.8% | 0.0042 | 0.3085 |
| 10% | Random | 38.5% | 9.0% | 95.6% | 0.3304 | 0.0051 |
| **15%** | **PTS (GBDT+Platt)** | **76.9%** | **74.3%** | **93.9%** | **0.0047** | **0.4850** |
| 15% | Heuristic | 76.9% | 15.2% | 94.8% | 0.2035 | 0.0115 |
| 15% | StaticAst | 69.2% | 12.2% | 94.7% | 0.1424 | 0.0059 |
| 15% | RecentFailureRate | 76.9% | 40.9% | 92.5% | 0.0042 | 0.3085 |
| 15% | Random | 46.2% | 13.4% | 93.2% | 0.3304 | 0.0051 |
| **20%** | **PTS (GBDT+Platt)** | **76.9%** | **90.1%** | **91.8%** | **0.0047** | **0.4850** |
| 20% | Heuristic | 69.2% | 19.1% | 92.5% | 0.2035 | 0.0115 |
| 20% | StaticAst | 69.2% | 19.1% | 92.1% | 0.1424 | 0.0059 |
| 20% | RecentFailureRate | 76.9% | 53.4% | 89.8% | 0.0042 | 0.3085 |
| 20% | Random | 46.2% | 17.9% | 90.8% | 0.3304 | 0.0051 |
| **30%** | **PTS (GBDT+Platt)** | **84.6%** | **94.0%** | **87.1%** | **0.0047** | **0.4850** |
| 30% | Heuristic | 84.6% | 22.4% | 87.8% | 0.2035 | 0.0115 |
| 30% | StaticAst | 69.2% | 25.1% | 87.3% | 0.1424 | 0.0059 |
| 30% | RecentFailureRate | 76.9% | 60.3% | 83.8% | 0.0042 | 0.3085 |
| 30% | Random | 61.5% | 26.0% | 85.8% | 0.3304 | 0.0051 |
| **50%** | **PTS (GBDT+Platt)** | **92.3%** | **96.4%** | **76.2%** | **0.0047** | **0.4850** |
| 50% | Heuristic | 69.2% | 33.4% | 77.8% | 0.2035 | 0.0115 |
| 50% | StaticAst | 76.9% | 36.4% | 77.9% | 0.1424 | 0.0059 |
| 50% | RecentFailureRate | 76.9% | 66.9% | 71.1% | 0.0042 | 0.3085 |
| 50% | Random | 61.5% | 39.1% | 75.5% | 0.3304 | 0.0051 |

At a **20% duration budget** (**91.8% wall-clock time saved**), **PTS (GBDT+Platt)** catches **90.1%** of all individual test failures (**0.4850 PR-AUC**, **95×** Random and **+57%** over `RecentFailureRate`). At a **50% budget** (**76.2% time saved**), PTS reaches **92.3% Commit FDR** and **96.4% Failure Recall**.

#### Probability Calibration Reliability

Across all 68,695 evaluated non-skip outcomes, raw GBDT achieves a `0.0041` Brier score vs. `0.0047` after Platt scaling (both >30× better than Heuristic `0.2035` and Random `0.3304`). Notice how Platt scaling eliminates raw GBDT's overconfident high-probability tail (`[0.3, 1.0]`, where raw margins predicted 37–81% failure probability on tests with 0–8% empirical failure rates).

![Probability Calibration Reliability Diagram — PyGithub/PyGithub](docs/benchmarks/calibration_reliability.png)

### Systems Latency & Footprint Benchmarks

Measured on **Apple M3 Pro** (macOS arm64, `rustc 1.98.1 (48a229cea 2026-09-01)`) against [`PyGithub/PyGithub`](https://github.com/PyGithub/PyGithub) (**1,232 discovered tests**, **93,736 stored outcomes**, `model_mode = gbdt`). Raw summary in [`docs/benchmarks/systems_bench.json`](docs/benchmarks/systems_bench.json).

| Metric | Value | Notes |
| :--- | ---: | :--- |
| Stripped release binary size (`pts`) | **8.22 MB** | Unstripped: 10.52 MB (static SQLite + `forust-ml` GBDT + Tree-sitter + `rmcp`) |
| `pts run --budget 15 --dry-run --json` (p50 / p95) | **1,897.2 ms / 2,025.0 ms** | Cold-start CLI over 15 runs (`git diff` + `pytest --collect-only` + SQLite + AST + GBDT+Platt) |
| `pts explain <test_id> --json` (p50 / p95) | **1,836.2 ms / 2,011.4 ms** | Cold-start CLI over 15 runs (16 features + top signals + Platt probability) |
| Pure `pts-ml` scoring latency (1,000 tests, p50 / p95) | **1.05 ms / 1.25 ms** | **949,067 tests/sec** (150-tree GBDT + Platt calibration via `bench_infer`) |

### Example `pts explain` Output (`PyGithub/PyGithub`)

```text
$ pts explain tests/PullRequest.py::PullRequest::testAttributes
Test:         tests/PullRequest.py::PullRequest::testAttributes
File:         tests/PullRequest.py
Model Mode:   gbdt
Fail Prob:    0.0000
Raw Margin:   -11.634332
Est Duration: 50 ms
Yield Score:  0.000000 prob/ms
Diff Files:   1

Top Contributing Signals:
  - path_jaccard_max         = 0.5000     (path token overlap (Jaccard=0.50) with modified file(s))
  - cochange_freq_max        = 0.4000     (frequently co-changed in git history (40% of modified file commits))
  - ast_min_hop              = 4.0        (test imports modified code within 4 AST hop(s))
  - touched_file_churn_30d   = 1.0        (modified file(s) had 1 commit(s) in the last 30 days)

16-Feature Breakdown:
   #  Feature                           Value
  --  -------------------------- ------------
   0  ast_min_hop                         4.0
   1  direct_test_modified                0.0
   2  path_jaccard_max                 0.5000
   3  same_dir_overlap                    0.0
   4  cofail_rate_max                  0.0455
   5  cochange_freq_max                0.4000
   6  ewma_fail_10                        0.0
   7  ewma_fail_50                        0.0
   8  runs_since_last_fail                NaN
   9  total_test_runs                    80.0
  10  median_duration_ms                 50.0
  11  flakiness_rate                      0.0
  12  diff_files_count                    1.0
  13  diff_lines_total                 1.0986
  14  diff_top_dirs_count                 1.0
  15  touched_file_churn_30d              1.0
```

### Reproducing Benchmarks

```bash
# 1. Collect pre-merge PR pytest runs from any public GitHub repository
python3 scripts/collect_gha_pytest.py \
  --repo PyGithub/PyGithub --workflow CI \
  --max-runs 100 --min-failures 25 \
  --out /tmp/pts-bench/pr_runs.jsonl

# 2. Initialize, ingest, evaluate, and render charts
cd /tmp/pts-bench-repo
pts init --history-limit 1000
pts import-runs /tmp/pts-bench/pr_runs.jsonl
pts eval --budgets 0.05,0.1,0.15,0.2,0.3,0.5 --output-json /path/to/pts/docs/benchmarks/eval_report.json

cd /path/to/pts
python3 scripts/plot_benchmarks.py --input docs/benchmarks/eval_report.json --out-dir docs/benchmarks --repo-label "PyGithub/PyGithub"

# 3. Measure release binary footprint, CLI latency, and pure pts-ml inference throughput
python3 scripts/bench_systems.py --repo-dir /tmp/pts-bench-repo --runs 15 --out-json docs/benchmarks/systems_bench.json
```

---

## Development

Run formatting, linter, Rust workspace tests, and Python collector unit tests:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python3 -m unittest scripts/test_collect_gha_pytest.py
```

---

## License

Licensed under the [MIT License](LICENSE).
