# failcast

Predictive `pytest` test selection for fast local loops and coding agents.

`failcast` is a single-binary Rust CLI and stdio MCP server that reads your working-tree `git diff` and predicts which `pytest` tests are likely to fail. It combines a local tabular GBDT model, a Tree-sitter Python import graph, and bundled SQLite to run the highest-yield subset inside your time budget with zero LLMs or cloud APIs.

[Install](#install) · [When to Use It](#when-to-use-it) · [How the ML Works](#how-the-ml-works) · [Benchmarks](#benchmarks) · [MCP Server](#mcp-server) · [Architecture](#architecture)

[![CI](https://github.com/AL3XPI/failcast/actions/workflows/ci.yml/badge.svg)](https://github.com/AL3XPI/failcast/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/failcast)](https://pypi.org/project/failcast/)
[![crates.io](https://img.shields.io/crates/v/failcast)](https://crates.io/crates/failcast)
[![Release](https://img.shields.io/github/v/release/AL3XPI/failcast?include_prereleases)](https://github.com/AL3XPI/failcast/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Failure Detection Rate vs. Time Budget — PyGithub/PyGithub](docs/benchmarks/pareto_fdr_vs_budget.png)

*Across 80 real PR runs on `PyGithub/PyGithub` (1,405 tests, 93,736 outcomes), a 20% time budget achieves 93.7% failure recall, saves 89.8% of wall-clock runtime, and reaches 0.689 PR-AUC.*

## Install

Run or install the prebuilt binary via PyPI (`uvx` or `pipx`, no Rust toolchain required) or compile from crates.io (Rust 1.88+, Edition 2024):

```bash
# Zero-install run via uvx
uvx failcast --help

# Install globally via pipx
pipx install failcast

# Build from crates.io
cargo install failcast
```

Prebuilt `v0.1.2` binaries (verified via [SHA256SUMS.txt](https://github.com/AL3XPI/failcast/releases/download/v0.1.2/SHA256SUMS.txt)):

[![macOS Apple Silicon](https://img.shields.io/badge/macOS_Apple_Silicon-failcast--v0.1.2--aarch64--apple--darwin.tar.gz-black?logo=apple)](https://github.com/AL3XPI/failcast/releases/download/v0.1.2/failcast-v0.1.2-aarch64-apple-darwin.tar.gz)

[![Linux x86_64](https://img.shields.io/badge/Linux_x86__64-failcast--v0.1.2--x86__64--unknown--linux--gnu.tar.gz-blue?logo=linux)](https://github.com/AL3XPI/failcast/releases/download/v0.1.2/failcast-v0.1.2-x86_64-unknown-linux-gnu.tar.gz)

[![Windows x64](https://img.shields.io/badge/Windows_x64-failcast--v0.1.2--x86__64--pc--windows--msvc.zip-0078D6?logo=windows)](https://github.com/AL3XPI/failcast/releases/download/v0.1.2/failcast-v0.1.2-x86_64-pc-windows-msvc.zip)

## When to Use It

`failcast` pays off on Python repositories whose full `pytest` suite takes 30 seconds or longer (roughly 200+ tests) and for coding agents firing tests in tight edit-verify loops. If your entire suite finishes in 3 seconds, running `pytest` directly is simpler.

Selection runs in two stages as your local SQLite store (`.failcast/store.db`) accumulates history. On Day 1, `failcast` ranks tests immediately with a zero-training heuristic built from AST import hop distance, direct test-file edits, path token overlap, git co-change frequency, and recent failure EWMA. Once `.failcast/store.db` reaches `min_runs_for_ml = 100` runs and `min_failures_for_ml = 25` failures (both configurable in `.failcast.toml`, or seeded upfront from CI logs via `failcast import-runs` or git replay via `failcast backfill`), `failcast train` fits the local ML model and takes over ranking.

## How the ML Works

The predictor is a **tabular Gradient Boosted Decision Tree (GBDT)** binary classifier (`forust-ml`, 150 trees, max depth 5) paired with **Platt scaling** (a 1D logistic sigmoid fit via damped Newton-Raphson) outputting calibrated failure probabilities. Training and inference stay local on CPU with zero LLMs, CNNs, or GNNs.

On each run, `tree-sitter` parses Python imports into a static file-level dependency graph and runs BFS to compute the shortest import hop distance (`ast_min_hop`) from each test file to the modified files. Raw per-file imports are cached incrementally in SQLite (`ast_file_cache`) by file size and nanosecond mtime so repeat runs only parse new or edited `.py` files while resolving graph edges accurately across the live tree. Those hop distances join 15 other tabular features (git co-change, historical co-failure rate, short- and medium-term failure EWMA, flakiness rate, median duration, diff size, and 30-day file churn) inside the GBDT. Candidates are packed greedily by yield (`calibrated_prob / max(duration_ms, 50)`), with a deterministic staleness-weighted exploration slice in live runs and inverse-propensity sample weighting during training to prevent test starvation and selection bias.

## CLI Quickstart

`pytest` must be on your `PATH` (or configured via `pytest_bin` in `.failcast.toml`). Add `.failcast/` to `.gitignore`, initialize the store, and run within a time budget:

```bash
# Initialize .failcast/store.db and mine recent git history for co-change and 30-day churn
failcast init --history-limit 500

# Run the full suite once to seed durations and baseline pass/fail history
failcast run --all

# Select and run highest-yield tests within a 15-second budget
failcast run --budget 15

# Preview selection without executing tests or writing runs to SQLite
failcast run --budget 15 --dry-run --json

# Bypass the SQLite discovery cache and force pytest --collect-only
failcast run --budget 15 --dry-run --refresh-tests --json

# Alternative selection modes
failcast run --top-k 20
failcast run --target-recall 0.95

# Train the GBDT + Platt model once history crosses thresholds (writes .failcast/model.json)
failcast train

# Inspect all 16 features, active model mode, calibrated probability, yield score, and top signals
failcast explain tests/test_calc.py::test_add_positive
failcast explain tests/test_calc.py::test_add_positive --json

# Replay stored runs chronologically against Random, StaticAst, and RecentFailureRate
failcast eval --budgets 0.1,0.2,0.3 --min-train-runs 20 --retrain-every 5 --output-json eval_report.json
```

When selected tests fail, `failcast run` parses the JUnit XML and commits every outcome to `.failcast/store.db` before propagating `pytest`'s non-zero exit code.

<details>
<summary>Configuration &amp; history import (.failcast.toml, backfill, JSONL)</summary>

Every key in `.failcast.toml` at your repository root is optional:

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

You can populate history by replaying recent commits inside an isolated temporary git worktree (leaving your working tree untouched) or by ingesting external CI logs from JSONL in a single SQLite transaction (`mode = 'backfill'`):

```bash
failcast backfill --commits 50 --ref HEAD
failcast import-runs ci_history.jsonl
```

```json
{"commit_sha":"a1b2c3d","started_at":1710000000,"diff_files":[{"path":"src/calc.py","lines_added":18,"lines_deleted":2}],"outcomes":[{"test_id":"tests/test_calc.py::test_divide_broken","file_path":"tests/test_calc.py","status":"fail","duration_ms":130}]}
```

</details>

## Benchmarks

Real pre-merge Pull Request runs from [`PyGithub/PyGithub`](https://github.com/PyGithub/PyGithub) (`CI` workflow, 2026-09-09 to 2026-11-20) were collected with [`scripts/collect_gha_pytest.py`](scripts/collect_gha_pytest.py). The dataset spans 80 PR runs (22 warmup runs and 58 strictly chronological evaluated runs with zero timestamp leakage), 19 failing PRs overall (13 failing runs and 335 test failures in the evaluation window), 1,405 unique `pytest` tests, and 93,736 outcomes (68,695 evaluated non-skip outcomes) with per-test durations taken directly from GitHub Actions JUnit XML (`pytest.xml`). Full evaluation metrics live in [`docs/benchmarks/eval_report.json`](docs/benchmarks/eval_report.json).

### 20% Duration Budget (`PyGithub/PyGithub`, 58 Evaluated Runs, 13 Failing Runs)

| Budget | Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **20%** | **Failcast (GBDT+Platt)** | **76.9%** | **93.7%** | **89.8%** | **0.0040** | **0.6894** |
| 20% | Heuristic | 61.5% | 14.6% | 92.8% | 0.2035 | 0.0115 |
| 20% | StaticAst | 61.5% | 15.2% | 92.3% | 0.1424 | 0.0059 |
| 20% | RecentFailureRate | 69.2% | 52.5% | 89.8% | 0.0042 | 0.3085 |
| 20% | Random | 38.5% | 13.4% | 91.1% | 0.3304 | 0.0051 |

### Systems Latency & Footprint

Measured on an Apple M3 Pro (macOS arm64, `rustc 1.98.1 (48a229cea 2026-09-01)`) against [`PyGithub/PyGithub`](https://github.com/PyGithub/PyGithub) with 1,232 discovered tests, 93,736 stored outcomes, and `model_mode = gbdt`. Raw measurements are saved in [`docs/benchmarks/systems_bench.json`](docs/benchmarks/systems_bench.json).

| Execution Path | Metric | Value | Notes |
| :--- | :--- | ---: | :--- |
| **Binary Footprint** | Stripped release binary (`failcast`) | **8.32 MB** | Unstripped: 10.66 MB (static SQLite + `forust-ml` + Tree-sitter + `rmcp`) |
| **Warm Rust Core** | `failcast run --budget 15 --dry-run --json` (p50 / p95) | **202.8 ms / 205.7 ms** | SQLite discovery + AST cache (**3.3× faster** than cold; diff + GBDT+Platt) |
| **Warm Rust Core** | `failcast explain <test_id> --json` (p50 / p95) | **135.2 ms / 139.3 ms** | SQLite discovery + AST cache (**4.7× faster** than cold; 16 features + signals) |
| **Cold `pytest` Discovery** | `failcast run --budget 15 --dry-run --refresh-tests --json` (p50 / p95) | **666.2 ms / 728.6 ms** | Includes external Python + `pytest --collect-only` startup across 1,232 tests |
| **Cold `pytest` Discovery** | `failcast explain <test_id> --refresh-tests --json` (p50 / p95) | **639.4 ms / 677.6 ms** | Includes external Python + `pytest --collect-only` startup |
| **Pure ML Core** | `failcast-ml` scoring latency (1,000 tests, p50 / p95) | **1.04 ms / 1.20 ms** | **963,507 tests/sec** (150-tree GBDT + Platt calibration via `bench_infer`) |

<details>
<summary>Full multi-budget table (5%–50%)</summary>

| Budget | Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **5%** | **Failcast (GBDT+Platt)** | **53.8%** | **26.6%** | **97.6%** | **0.0040** | **0.6894** |
| 5% | Heuristic | 46.2% | 6.3% | 98.8% | 0.2035 | 0.0115 |
| 5% | StaticAst | 38.5% | 5.1% | 98.8% | 0.1424 | 0.0059 |
| 5% | RecentFailureRate | 61.5% | 18.2% | 97.2% | 0.0042 | 0.3085 |
| 5% | Random | 30.8% | 3.0% | 97.9% | 0.3304 | 0.0051 |
| **10%** | **Failcast (GBDT+Platt)** | **69.2%** | **54.6%** | **94.9%** | **0.0040** | **0.6894** |
| 10% | Heuristic | 46.2% | 9.0% | 97.0% | 0.2035 | 0.0115 |
| 10% | StaticAst | 38.5% | 5.4% | 97.2% | 0.1424 | 0.0059 |
| 10% | RecentFailureRate | 69.2% | 30.1% | 94.9% | 0.0042 | 0.3085 |
| 10% | Random | 38.5% | 7.2% | 95.7% | 0.3304 | 0.0051 |
| **15%** | **Failcast (GBDT+Platt)** | **76.9%** | **80.0%** | **92.3%** | **0.0040** | **0.6894** |
| 15% | Heuristic | 53.8% | 12.2% | 95.0% | 0.2035 | 0.0115 |
| 15% | StaticAst | 53.8% | 11.0% | 94.9% | 0.1424 | 0.0059 |
| 15% | RecentFailureRate | 69.2% | 41.2% | 92.6% | 0.0042 | 0.3085 |
| 15% | Random | 38.5% | 10.4% | 93.4% | 0.3304 | 0.0051 |
| **20%** | **Failcast (GBDT+Platt)** | **76.9%** | **93.7%** | **89.8%** | **0.0040** | **0.6894** |
| 20% | Heuristic | 61.5% | 14.6% | 92.8% | 0.2035 | 0.0115 |
| 20% | StaticAst | 61.5% | 15.2% | 92.3% | 0.1424 | 0.0059 |
| 20% | RecentFailureRate | 69.2% | 52.5% | 89.8% | 0.0042 | 0.3085 |
| 20% | Random | 38.5% | 13.4% | 91.1% | 0.3304 | 0.0051 |
| **30%** | **Failcast (GBDT+Platt)** | **76.9%** | **94.3%** | **83.6%** | **0.0040** | **0.6894** |
| 30% | Heuristic | 69.2% | 16.7% | 88.5% | 0.2035 | 0.0115 |
| 30% | StaticAst | 61.5% | 19.1% | 87.9% | 0.1424 | 0.0059 |
| 30% | RecentFailureRate | 69.2% | 56.4% | 83.9% | 0.0042 | 0.3085 |
| 30% | Random | 53.8% | 20.6% | 86.3% | 0.3304 | 0.0051 |
| **50%** | **Failcast (GBDT+Platt)** | **76.9%** | **95.5%** | **71.3%** | **0.0040** | **0.6894** |
| 50% | Heuristic | 69.2% | 23.9% | 78.9% | 0.2035 | 0.0115 |
| 50% | StaticAst | 69.2% | 28.1% | 79.2% | 0.1424 | 0.0059 |
| 50% | RecentFailureRate | 69.2% | 62.7% | 70.6% | 0.0042 | 0.3085 |
| 50% | Random | 53.8% | 30.1% | 76.2% | 0.3304 | 0.0051 |

</details>

<details>
<summary>Probability Calibration Reliability</summary>

Across all 68,695 evaluated non-skip outcomes, Platt calibration with the support and log-loss guardrail pulls Brier score down from `0.0041` (raw GBDT) to `0.0040`, removing raw GBDT's overconfident `[0.3, 1.0]` tail bins.

![Probability Calibration Reliability Diagram — PyGithub/PyGithub](docs/benchmarks/calibration_reliability.png)

</details>

<details>
<summary>Implementation Notes</summary>

Spawning Python and `pytest --collect-only` on `PyGithub/PyGithub` burns roughly 500 ms before Rust starts work, and heavier Python dependency trees easily exceed 1.5 seconds. Caching both test discovery metadata (`repo_meta["discovery_fingerprint"]`) and per-file raw Python imports (`ast_file_cache` keyed by byte length and `mtime_ns`) inside `.failcast/store.db` drops warm end-to-end latency on `PyGithub/PyGithub` to 135 ms (`explain`) and 203 ms (`run --dry-run`), with 1.04 ms per 1,000 tests spent in GBDT+Platt scoring.

Wall-clock time saved exceeds `1 - budget` on fast unit suites because the selector clamps every test's effective duration to at least 50 ms (`MIN_EFF_DURATION_MS = 50.0`). Without that floor, sub-millisecond tests receive huge `prob / dur` ratios and crowd out slower tests that actually catch regressions. `PyGithub`'s unit tests have a 20 to 30 ms median runtime, so a 20% nominal budget picks fewer ultra-fast tests, saving 89.8% of actual wall-clock time while catching 93.7% of individual failures.

Real CI histories carry heavy class imbalance, often under 0.5% failing executions, so a fixed trailing 20% calibration window frequently hits zero or one failing run and squashes predicted probabilities. Expanding the calibration window backward down to a 60% training floor (while enforcing `max_train_ts < min_cal_ts`) until at least 5 positives across at least 2 failing runs appear, and falling back to an identity sigmoid (`a = 1.0, b = 0.0`) when support is thin or log-loss worsens, prevents calibration collapse.

</details>

<details>
<summary>Example <code>failcast explain</code> Output (PyGithub/PyGithub)</summary>

```text
$ failcast explain tests/PullRequest.py::PullRequest::testAttributes
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

</details>

<details>
<summary>Reproducing Benchmarks</summary>

```bash
# Collect pre-merge PR pytest runs from any public GitHub repository
python3 scripts/collect_gha_pytest.py \
  --repo PyGithub/PyGithub --workflow CI \
  --max-runs 100 --min-failures 25 \
  --out ./bench-data/pr_runs.jsonl

# Initialize, import, evaluate, and plot against a local checkout of the target repo
git clone https://github.com/PyGithub/PyGithub ../PyGithub
(cd ../PyGithub \
  && failcast init --history-limit 1000 \
  && failcast import-runs ../failcast/bench-data/pr_runs.jsonl \
  && failcast eval --budgets 0.05,0.1,0.15,0.2,0.3,0.5 --output-json ../failcast/docs/benchmarks/eval_report.json)

python3 scripts/plot_benchmarks.py --input docs/benchmarks/eval_report.json --out-dir docs/benchmarks --repo-label "PyGithub/PyGithub"

# Measure release binary footprint, CLI latency, and pure failcast-ml inference throughput
python3 scripts/bench_systems.py --repo-dir ../PyGithub --runs 15 --out-json docs/benchmarks/systems_bench.json
```

</details>

## MCP Server

`failcast mcp` exposes a local [Model Context Protocol](https://modelcontextprotocol.io) server over `stdio` via `rmcp`. JSON-RPC 2.0 frames own `stdout`, while tracing diagnostics route to `stderr`.

| Tool | Description |
| :--- | :--- |
| `select_tests` | Dry-run test selection for the current working-tree diff (`budget_seconds`, `top_k`, `target_recall`, `refresh_tests`). |
| `run_fast_tests` | Select and execute tests within a budget or top-k limit, persist outcomes to `.failcast/store.db`, and return pass/fail counts with bounded tracebacks (`budget_seconds`, `top_k`, `run_all`, `refresh_tests`). |
| `explain_selection` | Return all 16 features, active model mode, calibrated failure probability, yield score, and top contributing signals for a `test_id` (`test_id`, `refresh_tests`). |

Configure your MCP client (Claude Desktop, Cursor, Claude Code, Antigravity) with `cwd` pointing to your Python repository using either the installed binary or `uvx`:

```json
{
  "mcpServers": {
    "failcast": {
      "command": "failcast",
      "args": ["mcp"],
      "cwd": "/path/to/your/python-repo"
    }
  }
}
```

```json
{
  "mcpServers": {
    "failcast": {
      "command": "uvx",
      "args": ["failcast", "mcp"],
      "cwd": "/path/to/your/python-repo"
    }
  }
}
```

## Architecture

| Crate | Responsibility |
| :--- | :--- |
| `failcast-core` | Pure domain types (`TestCase`, `DiffSummary`, `HistorySnapshot`, `Config`, `TestRunner`, `FailcastError`) with zero I/O or subprocesses. |
| `failcast-store` | Bundled SQLite store in WAL mode, schema migrations (`v1`–`v3`), atomic run/batch writes, incremental `ast_file_cache`, and point-in-time (`< as_of_ts`) history queries. |
| `failcast-collectors` | Working-tree `git diff` and `git log` mining, isolated `TempGitWorktree` for commit backfills, and incremental Tree-sitter Python AST import graph building. |
| `failcast-ml` | 16-feature extractor, cold-start heuristic ranker, `forust-ml` GBDT trainer, Newton-Raphson Platt calibrator, and deterministic cost-aware selector. |
| `failcast-runners` | `TestRunner` implementation for `pytest` with JUnit XML parsing and bounded failure tracebacks. |
| `failcast-eval` | Chronological walk-forward evaluator comparing Failcast against `Random`, `StaticAst`, and `RecentFailureRate` baselines. |
| `failcast` (`crates/failcast-cli`) | The `failcast` binary: shared `service.rs`, Clap subcommands, JSONL CI importer, and `rmcp` stdio MCP server. |

<details>
<summary>Dependency graph</summary>

```text
                    +-------------------------------------------+
                    |        failcast (crates/failcast-cli)     |
                    |   Clap CLI    |  service.rs  |  rmcp MCP  |
                    +---------------+------+-------+------------+
                                           |
          +--------------------------------+--------------------------------+
          |                                |                                |
          v                                v                                v
+--------------------+         +-----------------------+         +---------------------+
| failcast-collectors|         |      failcast-ml      |         |  failcast-runners   |
| - git diff/history |         | - 16 pure features    |         | - PytestRunner      |
| - tree-sitter Py   |         | - Cold-start ranker   |         | - JUnit XML parser  |
|   AST import graph |         | - forust-ml GBDT      |         | - Bounded tracebacks|
| - TempGitWorktree  |         | - Platt calibration   |         +---------------------+
+---------+----------+         | - Cost-aware selector |                    |
          |                    +-----------+-----------+                    |
          +----------------------+         |         +----------------------+
                                 v         v         v
                    +--------------------+   +--------------------+
                    |   failcast-store   |   |   failcast-eval    |
                    | - Bundled SQLite   |   | - Walk-forward     |
                    | - WAL + migrations |   | - Random/AST/EWMA  |
                    | - As-of snapshots  |   |   baselines        |
                    +----------+---------+   +----------+---------+
                               |                        |
                               +-----------+------------+
                                           v
                              +-------------------------+
                              |      failcast-core      |
                              | Pure types, Config,     |
                              | TestRunner trait, Error |
                              +-------------------------+
```

</details>

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python3 -m unittest scripts/test_collect_gha_pytest.py
```

## License

MIT. See [LICENSE](LICENSE).

<!-- mcp-name: io.github.AL3XPI/failcast -->
