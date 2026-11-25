# pts

[![CI](https://github.com/AL3XPI/pts/actions/workflows/ci.yml/badge.svg)](https://github.com/AL3XPI/pts/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/AL3XPI/pts?include_prereleases)](https://github.com/AL3XPI/pts/releases)
![Rust](https://img.shields.io/badge/Rust-2024%20%7C%20rustc%201.98.1-dea584?logo=rust&logoColor=white)
![forust-ml](https://img.shields.io/badge/forust--ml-0.8.1-blue)
![rmcp](https://img.shields.io/badge/rmcp-3.11.0-blue)
![tree-sitter](https://img.shields.io/badge/tree--sitter-0.25.10-blue)
![SQLite](https://img.shields.io/badge/SQLite-3.49.1%20%28bundled%29-003b57?logo=sqlite&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`pts` is a local Rust CLI and stdio MCP server that reads your working-tree `git diff`, predicts which `pytest` tests are likely to fail, runs the highest-yield ones inside a time or count budget, and stores every outcome in SQLite so future runs get sharper.

Running a thousand Python tests on every small edit burns minutes, and picking test files by hand misses transitive regressions. Everything here compiles into one Rust binary (Edition 2024). SQLite is statically linked through bundled `rusqlite`, gradient-boosted trees train and score in pure Rust with `forust-ml` plus Platt calibration, and `tree-sitter` builds a file-level Python import graph directly from source.

The selector ranks candidate tests by calibrated failure probability per millisecond (`prob / eff_ms`). A small deterministic exploration pass adds stale tests on top of the greedy budget so neglected parts of the suite stay measured, and inverse-propensity weighting corrects training samples for selection bias. Training and evaluation splits follow chronological run boundaries (`max_train_ts < min_cal_ts < min_eval_ts`) to keep future outcomes out of earlier feature snapshots. The same internal service layer drives the CLI (`pts run`, `pts explain`) and the `rmcp` stdio MCP server (`pts mcp`).

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

### Crate Layout

| Crate | Responsibility |
| :--- | :--- |
| `pts-core` | Pure domain types (`TestCase`, `DiffSummary`, `HistorySnapshot`, `Config`, `TestRunner`, `PtsError`). Zero I/O or subprocesses. |
| `pts-store` | Bundled SQLite store in WAL mode, schema migrations, atomic run/batch writes, and point-in-time (`< as_of_ts`) history queries. |
| `pts-collectors` | Working-tree `git diff` and `git log` mining, isolated `TempGitWorktree` for commit backfills, and a Tree-sitter Python file-level import graph. |
| `pts-ml` | 16-feature extractor, cold-start heuristic ranker, `forust-ml` GBDT trainer, Newton-Raphson Platt calibrator, and deterministic cost-aware selector. |
| `pts-runners` | `TestRunner` implementation for `pytest` with JUnit XML parsing and bounded failure tracebacks. |
| `pts-eval` | Chronological walk-forward evaluator comparing PTS against `Random`, `StaticAst`, and `RecentFailureRate` baselines. |
| `pts-cli` | The `pts` binary: shared `service.rs`, Clap subcommands, JSONL CI importer, and `rmcp` stdio MCP server. |

## Installation

You need Rust 1.88+ (Edition 2024, tested on `rustc 1.98.1`) and `pytest` on your `PATH` (or configured in `.pts.toml`).

```bash
cargo install --path crates/pts-cli
```

Check the binary:

```bash
pts --help
```

## CLI Quickstart

### Initialize a repository

`pts init` creates `.pts/store.db` and walks recent git history to compute file co-change frequencies and 30-day churn. Put `.pts/` in your repo's `.gitignore`.

```bash
pts init --history-limit 500
```

### Run tests with predictive selection

Run the full suite once to seed durations and pass/fail history, then switch to budgeted or count-limited runs.

```bash
# Run the full suite once to seed timing and baseline outcomes
pts run --all

# Select and run highest-value tests within a 15-second budget
pts run --budget 15

# Preview selection without executing tests or writing runs to SQLite (uses SQLite discovery cache)
pts run --budget 15 --dry-run --json

# Force fresh pytest --collect-only discovery (bypasses the SQLite cache)
pts run --budget 15 --dry-run --refresh-tests --json

# Alternative selection modes
pts run --top-k 20
pts run --target-recall 0.95
```

When tests fail, `pts run` still parses the JUnit XML and commits every outcome to `.pts/store.db` before returning `pytest`'s non-zero exit code.

### Train the GBDT + Platt model

Until enough history accumulates (`min_runs_for_ml = 100` and `min_failures_for_ml = 25` by default, both editable in `.pts.toml`), `pts` ranks with a cold-start heuristic. Once you pass those thresholds, train the tree model:

```bash
pts train
```

This writes `.pts/model.json` and stores the calibration parameters and Brier score metadata in SQLite.

### Inspect why a test ranked where it did

`pts explain` prints all 16 extracted features, the active model mode (`gbdt` or `heuristic`), raw GBDT margin, calibrated failure probability, yield score, and top contributing signals for a single test.

```bash
pts explain tests/test_calc.py::test_add_positive
pts explain tests/test_calc.py::test_add_positive --json
```

### Populate history from git commits or CI logs

Waiting for 100 local runs takes time. You can replay recent commits inside an isolated temporary worktree so your working tree stays untouched:

```bash
pts backfill --commits 50 --ref HEAD
```

You can also load external CI or PR failure logs from a JSONL file straight into `.pts/store.db` (`mode = 'backfill'`). Each line is one historical run ingested in one transaction:

```bash
pts import-runs ci_history.jsonl
```

```json
{"commit_sha":"a1b2c3d","started_at":1710000000,"diff_files":[{"path":"src/calc.py","lines_added":18,"lines_deleted":2}],"outcomes":[{"test_id":"tests/test_calc.py::test_divide_broken","file_path":"tests/test_calc.py","status":"fail","duration_ms":130}]}
```

### Chronological walk-forward evaluation

`pts eval` replays stored runs in strict timestamp order and scores PTS against `Random`, `StaticAst`, and `RecentFailureRate` across multiple time budgets while checking for timestamp leakage on every step:

```bash
pts eval --budgets 0.1,0.2,0.3 --min-train-runs 20 --retrain-every 5 --output-json eval_report.json
```

## Configuration (`.pts.toml`)

Every key is optional. Copy `.pts.toml.example` to `.pts.toml` at your repository root to override any default:

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

## MCP Server (`pts mcp`)

`pts mcp` runs a local [Model Context Protocol](https://modelcontextprotocol.io) server over `stdio` using `rmcp`. Only JSON-RPC 2.0 frames go to `stdout`; tracing logs and subprocess diagnostics stay off `stdout`.

### Exposed Tools

| Tool | Description |
| :--- | :--- |
| `select_tests` | Dry-run test selection for the current working-tree diff (`budget_seconds`, `top_k`, `target_recall`, `refresh_tests`). |
| `run_fast_tests` | Select and execute tests within a budget or top-k limit, persist outcomes to `.pts/store.db`, and return structured pass/fail counts with bounded tracebacks (`budget_seconds`, `top_k`, `run_all`, `refresh_tests`). |
| `explain_selection` | Return all 16 features, active model mode, calibrated failure probability, yield score, and top contributing signals for a specific `test_id` (`test_id`, `refresh_tests`). |

### Client Configuration (Stdio)

Point your MCP client (Claude Desktop, Cursor, Claude Code, Antigravity, etc.) at `pts mcp` with `cwd` set to the Python repo you are working in:

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

## Benchmarks

### Empirical Walk-Forward Evaluation (`PyGithub/PyGithub`)

I collected real pre-merge Pull Request runs from [`PyGithub/PyGithub`](https://github.com/PyGithub/PyGithub) (`CI` workflow) spanning 2026-09-09 to 2026-11-20 using [`scripts/collect_gha_pytest.py`](scripts/collect_gha_pytest.py). The dataset contains 80 PR runs total: 22 warmup runs followed by 58 strictly chronological evaluated runs, with zero timestamp leakage verified on every step. Across those 80 runs there are 19 failing PRs, leaving 13 failing runs and 335 individual test failures inside the 58-run evaluation window. That covers 1,405 unique `pytest` tests and 93,736 total outcomes (68,695 evaluated non-skip outcomes). Every single per-test duration comes straight from GitHub Actions JUnit XML (`pytest.xml`) workflow artifacts (`0.0%` estimated; 2 zero-pass environment/collection-crash runs excluded).

Evaluation trains a 150-tree depth-5 `forust-ml` GBDT with chronological Platt calibration every 5 runs (`gbdt_steps = 58 / 58`). Platt calibration requires at least 5 positives across at least 2 failing runs and falls back to an identity sigmoid when support is thin or log-loss worsens. Offline walk-forward evaluation sets `explore_fraction = 0.0` so all five strategies compete strictly on ranking and cost-aware packing, while live CLI and MCP runs keep anti-starvation exploration enabled. Full JSON output lives in [`docs/benchmarks/eval_report.json`](docs/benchmarks/eval_report.json).

![Failure Detection Rate vs. Time Budget — PyGithub/PyGithub](docs/benchmarks/pareto_fdr_vs_budget.png)

#### Chronological Walk-Forward Comparison (`PyGithub/PyGithub`, 58 Evaluated Runs, 13 Failing Runs)

| Budget | Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **5%** | **PTS (GBDT+Platt)** | **53.8%** | **26.6%** | **97.6%** | **0.0040** | **0.6894** |
| 5% | Heuristic | 46.2% | 6.3% | 98.8% | 0.2035 | 0.0115 |
| 5% | StaticAst | 38.5% | 5.1% | 98.8% | 0.1424 | 0.0059 |
| 5% | RecentFailureRate | 61.5% | 18.2% | 97.2% | 0.0042 | 0.3085 |
| 5% | Random | 30.8% | 3.0% | 97.9% | 0.3304 | 0.0051 |
| **10%** | **PTS (GBDT+Platt)** | **69.2%** | **54.6%** | **94.9%** | **0.0040** | **0.6894** |
| 10% | Heuristic | 46.2% | 9.0% | 97.0% | 0.2035 | 0.0115 |
| 10% | StaticAst | 38.5% | 5.4% | 97.2% | 0.1424 | 0.0059 |
| 10% | RecentFailureRate | 69.2% | 30.1% | 94.9% | 0.0042 | 0.3085 |
| 10% | Random | 38.5% | 7.2% | 95.7% | 0.3304 | 0.0051 |
| **15%** | **PTS (GBDT+Platt)** | **76.9%** | **80.0%** | **92.3%** | **0.0040** | **0.6894** |
| 15% | Heuristic | 53.8% | 12.2% | 95.0% | 0.2035 | 0.0115 |
| 15% | StaticAst | 53.8% | 11.0% | 94.9% | 0.1424 | 0.0059 |
| 15% | RecentFailureRate | 69.2% | 41.2% | 92.6% | 0.0042 | 0.3085 |
| 15% | Random | 38.5% | 10.4% | 93.4% | 0.3304 | 0.0051 |
| **20%** | **PTS (GBDT+Platt)** | **76.9%** | **93.7%** | **89.8%** | **0.0040** | **0.6894** |
| 20% | Heuristic | 61.5% | 14.6% | 92.8% | 0.2035 | 0.0115 |
| 20% | StaticAst | 61.5% | 15.2% | 92.3% | 0.1424 | 0.0059 |
| 20% | RecentFailureRate | 69.2% | 52.5% | 89.8% | 0.0042 | 0.3085 |
| 20% | Random | 38.5% | 13.4% | 91.1% | 0.3304 | 0.0051 |
| **30%** | **PTS (GBDT+Platt)** | **76.9%** | **94.3%** | **83.6%** | **0.0040** | **0.6894** |
| 30% | Heuristic | 69.2% | 16.7% | 88.5% | 0.2035 | 0.0115 |
| 30% | StaticAst | 61.5% | 19.1% | 87.9% | 0.1424 | 0.0059 |
| 30% | RecentFailureRate | 69.2% | 56.4% | 83.9% | 0.0042 | 0.3085 |
| 30% | Random | 53.8% | 20.6% | 86.3% | 0.3304 | 0.0051 |
| **50%** | **PTS (GBDT+Platt)** | **76.9%** | **95.5%** | **71.3%** | **0.0040** | **0.6894** |
| 50% | Heuristic | 69.2% | 23.9% | 78.9% | 0.2035 | 0.0115 |
| 50% | StaticAst | 69.2% | 28.1% | 79.2% | 0.1424 | 0.0059 |
| 50% | RecentFailureRate | 69.2% | 62.7% | 70.6% | 0.0042 | 0.3085 |
| 50% | Random | 53.8% | 30.1% | 76.2% | 0.3304 | 0.0051 |

At a 20% duration budget (89.8% wall-clock time saved), PTS catches 93.7% of individual test failures with a 0.6894 PR-AUC (135× Random, 60× Heuristic, 2.23× `RecentFailureRate`). Every strategy's Commit FDR curve stays monotonically non-decreasing as the budget grows.

#### Probability Calibration Reliability

Across all 68,695 evaluated non-skip outcomes, Platt calibration with the support and log-loss guardrail pulls Brier score down from `0.0041` (raw GBDT) to `0.0040`, a 3.5% drop and the lowest of the five strategies, while wiping out raw GBDT's overconfident `[0.3, 1.0]` tail bins.

![Probability Calibration Reliability Diagram — PyGithub/PyGithub](docs/benchmarks/calibration_reliability.png)

### Systems Latency & Footprint

Measured on an Apple M3 Pro (macOS arm64, `rustc 1.98.1 (48a229cea 2026-09-01)`) against [`PyGithub/PyGithub`](https://github.com/PyGithub/PyGithub) with 1,232 discovered tests, 93,736 stored outcomes, and `model_mode = gbdt`. Raw measurements are saved in [`docs/benchmarks/systems_bench.json`](docs/benchmarks/systems_bench.json).

| Execution Path | Metric | Value | Notes |
| :--- | :--- | ---: | :--- |
| **Binary Footprint** | Stripped release binary (`pts`) | **8.26 MB** | Unstripped: 10.55 MB (static SQLite + `forust-ml` GBDT + Tree-sitter + `rmcp`) |
| **Warm Rust Core** | `pts run --budget 15 --dry-run --json` (p50 / p95) | **398.3 ms / 503.1 ms** | Cached SQLite discovery (**2.2× faster** than cold; `git diff` + SQLite + AST + GBDT+Platt) |
| **Warm Rust Core** | `pts explain <test_id> --json` (p50 / p95) | **357.2 ms / 364.6 ms** | Cached SQLite discovery (**2.4× faster** than cold; 16 features + signals + Platt prob) |
| **Cold `pytest` Discovery** | `pts run --budget 15 --dry-run --refresh-tests --json` (p50 / p95) | **874.3 ms / 925.3 ms** | Includes external Python + `pytest --collect-only` startup across 1,232 tests |
| **Cold `pytest` Discovery** | `pts explain <test_id> --refresh-tests --json` (p50 / p95) | **865.4 ms / 899.7 ms** | Includes external Python + `pytest --collect-only` startup |
| **Pure ML Core** | `pts-ml` scoring latency (1,000 tests, p50 / p95) | **1.10 ms / 1.50 ms** | **905,182 tests/sec** (150-tree GBDT + Platt calibration via `bench_infer`) |

#### Implementation Notes

Spawning Python and `pytest --collect-only` on `PyGithub/PyGithub` burns roughly 500 ms before Rust does any work, and heavier Python dependency graphs easily hit 1.5 to 2 seconds. Hashing test and pytest config file metadata into SQLite (`repo_meta["discovery_fingerprint"]`) keeps repeat `pts run` and `pts explain` calls inside Rust at 357 to 398 ms p50 end-to-end (with 1.10 ms per 1,000 tests spent in GBDT+Platt scoring) and invalidates automatically whenever a test or pytest config file changes.

Wall-clock time saved exceeds `1 - budget` on fast unit suites because the selector clamps every test's effective duration to at least 50 ms (`MIN_EFF_DURATION_MS = 50.0`). Without that floor, sub-millisecond tests get huge `prob / dur` yield ratios and crowd out slower tests that actually break. `PyGithub`'s unit tests have a 20 to 30 ms median runtime, so a 20% nominal budget picks fewer ultra-fast tests than `20% / 25ms`, saving 89.8% of actual wall-clock time while catching 93.7% of individual failures.

Real CI histories have heavy class imbalance, often under 0.5% failing executions, so a fixed trailing 20% calibration window frequently hits zero or one failing run and squashes predicted probabilities. Expanding the calibration window backward down to a 60% training floor (while keeping `max_train_ts < min_cal_ts`) until at least 5 positives across at least 2 failing runs appear, and falling back to an identity sigmoid (`a = 1.0, b = 0.0`) when support is thin or log-loss gets worse, keeps calibration from collapsing and drops Brier score from `0.0041` to `0.0040`.

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
# Collect pre-merge PR pytest runs from any public GitHub repository
python3 scripts/collect_gha_pytest.py \
  --repo PyGithub/PyGithub --workflow CI \
  --max-runs 100 --min-failures 25 \
  --out ./bench-data/pr_runs.jsonl

# Initialize, import, evaluate, and plot against a local checkout of the target repo
git clone https://github.com/PyGithub/PyGithub ../PyGithub
(cd ../PyGithub \
  && pts init --history-limit 1000 \
  && pts import-runs ../pts/bench-data/pr_runs.jsonl \
  && pts eval --budgets 0.05,0.1,0.15,0.2,0.3,0.5 --output-json ../pts/docs/benchmarks/eval_report.json)

python3 scripts/plot_benchmarks.py --input docs/benchmarks/eval_report.json --out-dir docs/benchmarks --repo-label "PyGithub/PyGithub"

# Measure release binary footprint, CLI latency, and pure pts-ml inference throughput
python3 scripts/bench_systems.py --repo-dir ../PyGithub --runs 15 --out-json docs/benchmarks/systems_bench.json
```

## Development

Run formatting, linter, Rust workspace tests, and Python collector unit tests:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python3 -m unittest scripts/test_collect_gha_pytest.py
```

## License

MIT. See [LICENSE](LICENSE).
