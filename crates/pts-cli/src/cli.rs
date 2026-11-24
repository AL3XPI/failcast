use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use pts_collectors::{
    ImportGraphBuilder, PythonImportGraphBuilder, extract_working_diff, is_git_repo,
    mine_commit_history,
};
use pts_core::{
    Config, DiffSummary, HistorySnapshot, PtsError, RunMode, RunnerResult, SelectionReason,
    TestCase, TestRunner, TestStatus,
};
use pts_ml::{
    CandidateTest, SelectedTest, SelectionStrategy, extract_features, score_heuristic, select_tests,
};
use pts_runners::{PytestRunner, normalize_posix_path};
use pts_store::{RunRecord, Store, TestOutcomeRecord};
use serde::Serialize;
use tracing::{info, warn};
use uuid::Uuid;

use crate::mcp;

const DEFAULT_UNSEEN_STALENESS_DAYS: f64 = 30.0;

#[derive(Debug, Parser)]
#[command(
    name = "pts",
    version,
    about = "Predictive Test Selector — run the tests most likely to fail"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Initialize .pts/ state directory, SQLite store, and mine git history.
    Init(InitArgs),
    /// Select and execute tests, persisting outcomes to .pts/store.db.
    Run(RunArgs),
    /// Train the GBDT + Platt calibration model (Milestone 3).
    Train,
    /// Explain the ranking features and score for a specific test (Milestone 4).
    Explain { test_id: String },
    /// Run chronological walk-forward evaluation against baselines (Milestone 3).
    Eval {
        #[arg(long)]
        output_json: PathBuf,
    },
    /// Start the local MCP server over stdio (Milestone 4).
    Mcp,
}

#[derive(Debug, Args)]
pub struct InitArgs {
    /// Maximum number of git commits to mine during initialization (defaults to config value, 500).
    #[arg(long)]
    pub history_limit: Option<usize>,
}

#[derive(Debug, Args)]
#[command(group(
    clap::ArgGroup::new("selection_mode")
        .required(true)
        .args(["all", "budget", "top_k", "target_recall"])
))]
pub struct RunArgs {
    /// Run the entire discovered test suite.
    #[arg(long)]
    pub all: bool,

    /// Time budget in seconds.
    #[arg(long, value_name = "SECS")]
    pub budget: Option<u64>,

    /// Maximum number of tests to select.
    #[arg(long, value_name = "N")]
    pub top_k: Option<usize>,

    /// Target failure recall in (0, 1).
    #[arg(long, value_name = "0..1")]
    pub target_recall: Option<f64>,

    /// Print the selected tests without executing them or writing to the database.
    #[arg(long)]
    pub dry_run: bool,

    /// Emit machine-readable JSON output to stdout.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct DryRunSelectedItem {
    test_id: String,
    predicted_fail_prob: Option<f64>,
    est_duration_ms: Option<u64>,
    reason: SelectionReason,
    sampling_propensity: f64,
}

#[derive(Debug, Serialize)]
struct DryRunJsonReport {
    mode: String,
    model_mode: String,
    dry_run: bool,
    diff_files_count: usize,
    total_discovered_tests: usize,
    selected_count: usize,
    estimated_duration_ms: u64,
    selected: Vec<DryRunSelectedItem>,
}

#[derive(Debug, Serialize)]
struct RunJsonReport {
    run_id: String,
    mode: RunMode,
    model_mode: String,
    exit_code: i32,
    total_duration_ms: u64,
    passed: usize,
    failed: usize,
    errored: usize,
    skipped: usize,
    total: usize,
    failures: Vec<FailureSummary>,
}

#[derive(Debug, Serialize)]
struct FailureSummary {
    test_id: String,
    duration_ms: u64,
    failure_message: Option<String>,
}

/// Executes the parsed CLI command and returns the desired process exit code.
pub fn execute(cli: Cli, repo_root: &Path) -> Result<i32> {
    match cli.command {
        Command::Init(args) => {
            cmd_init(repo_root, args)?;
            Ok(0)
        }
        Command::Run(args) => cmd_run(repo_root, args),
        Command::Train => bail!("`pts train` is planned for Milestone 3"),
        Command::Explain { .. } => bail!("`pts explain` is planned for Milestone 4"),
        Command::Eval { .. } => bail!("`pts eval` is planned for Milestone 3"),
        Command::Mcp => {
            mcp::run_mcp()?;
            Ok(0)
        }
    }
}

fn cmd_init(repo_root: &Path, args: InitArgs) -> Result<()> {
    let config = Config::load(repo_root)?;
    let history_limit = args.history_limit.unwrap_or(config.history_commit_limit);

    let state_dir = repo_root.join(".pts");
    std::fs::create_dir_all(&state_dir)
        .with_context(|| format!("failed to create {}", state_dir.display()))?;

    let db_path = state_dir.join("store.db");
    let mut store = Store::open(&db_path)?;
    store.migrate()?;

    let mined_commits = if is_git_repo(repo_root) {
        let commits = mine_commit_history(repo_root, history_limit)?;
        store.upsert_commits(&commits)?
    } else {
        warn!(
            repo = %repo_root.display(),
            "repository is not a git working tree; skipped git commit history mining"
        );
        0
    };

    check_gitignore_warning(repo_root);
    info!(
        db = %db_path.display(),
        history_limit,
        mined_commits,
        "initialized PTS store"
    );
    println!(
        "Initialized PTS store at {} (mined {} commit(s))",
        db_path.display(),
        mined_commits
    );
    Ok(())
}

fn check_gitignore_warning(repo_root: &Path) {
    let gitignore_path = repo_root.join(".gitignore");
    let has_pts_entry = std::fs::read_to_string(&gitignore_path)
        .ok()
        .is_some_and(|content| {
            content.lines().any(|line| {
                let trimmed = line.trim();
                trimmed == ".pts" || trimmed == ".pts/" || trimmed == "/.pts" || trimmed == "/.pts/"
            })
        });

    if !has_pts_entry {
        warn!("`.pts/` is not listed in .gitignore; consider adding `.pts/` to .gitignore");
    }
}

fn cmd_run(repo_root: &Path, args: RunArgs) -> Result<i32> {
    if let Some(recall) = args.target_recall
        && (!recall.is_finite() || recall <= 0.0 || recall >= 1.0)
    {
        bail!("--target-recall must be strictly between 0.0 and 1.0 (got {recall})");
    }

    let config = Config::load(repo_root)?;
    let db_path = repo_root.join(".pts").join("store.db");
    if !args.dry_run && !db_path.is_file() {
        return Err(PtsError::NotInitialized.into());
    }

    let diff = if is_git_repo(repo_root) {
        extract_working_diff(repo_root)?
    } else {
        warn!(
            repo = %repo_root.display(),
            "repository is not a git working tree; using empty diff"
        );
        DiffSummary::from_files(None, Vec::new())
    };

    let runner = PytestRunner::from_config(&config);
    let discovered = runner.discover(repo_root)?;
    let now_ts = current_unix_ts();

    let (mode, mode_label, model_mode, budget_ms, selected_plan): (
        RunMode,
        &'static str,
        &'static str,
        Option<i64>,
        Option<Vec<SelectedTest>>,
    ) = if args.all {
        (RunMode::All, "all", "full", None, None)
    } else {
        let history = if db_path.is_file() {
            let store = Store::open(&db_path)?;
            store.load_history_snapshot(now_ts)?
        } else {
            HistorySnapshot::default()
        };

        let graph = PythonImportGraphBuilder::new().build(repo_root)?;
        let modified_paths: Vec<&str> = diff.files.iter().map(|f| f.path.as_str()).collect();

        let candidates: Vec<CandidateTest> = discovered
            .iter()
            .map(|tc| {
                let ast_min_hop = graph.min_hop_to_any(&tc.file_path, &modified_paths);
                let features = extract_features(tc, &diff, &history, ast_min_hop);
                let prob = score_heuristic(&features);
                let test_hist = history.test_history.get(&tc.test_id);
                let est_duration_ms = test_hist.and_then(|h| h.median_duration_ms);
                let days_since_last_run = test_hist
                    .and_then(|h| h.last_run_ts)
                    .map(|ts| ((now_ts - ts).max(0) as f64) / 86_400.0)
                    .unwrap_or(DEFAULT_UNSEEN_STALENESS_DAYS);
                CandidateTest {
                    test_id: tc.test_id.clone(),
                    prob,
                    est_duration_ms,
                    days_since_last_run,
                }
            })
            .collect();

        let (strategy, run_mode, label, budget_ms_opt) = if let Some(secs) = args.budget {
            let ms = secs.saturating_mul(1000);
            (
                SelectionStrategy::Budget { budget_ms: ms },
                RunMode::Budget,
                "budget",
                Some(ms as i64),
            )
        } else if let Some(k) = args.top_k {
            (SelectionStrategy::TopK { k }, RunMode::TopK, "top_k", None)
        } else if let Some(recall) = args.target_recall {
            (
                SelectionStrategy::TargetRecall {
                    target_recall: recall,
                    max_budget_ms: None,
                },
                RunMode::Budget,
                "target_recall",
                None,
            )
        } else {
            unreachable!("clap ArgGroup guarantees one selection mode is set");
        };

        let selected = select_tests(
            &candidates,
            strategy,
            &diff.diff_fingerprint,
            history.median_repo_test_ms,
            &config,
            SelectionReason::Heuristic,
        );
        (run_mode, label, "heuristic", budget_ms_opt, Some(selected))
    };

    if args.dry_run {
        emit_dry_run_report(
            mode_label,
            model_mode,
            diff.files.len(),
            &discovered,
            selected_plan.as_deref(),
            args.json,
        )?;
        return Ok(0);
    }

    let mut store = Store::open(&db_path)?;
    store.migrate()?;

    let run_id = Uuid::new_v4().to_string();
    let started_at = current_unix_ts();

    let result = match &selected_plan {
        None => runner.execute(repo_root, None)?,
        Some(plan) if plan.is_empty() => RunnerResult {
            exit_code: 0,
            total_duration_ms: 0,
            outcomes: Vec::new(),
        },
        Some(plan) => {
            let selected_ids: Vec<String> = plan.iter().map(|s| s.test_id.clone()).collect();
            runner.execute(repo_root, Some(&selected_ids))?
        }
    };
    let finished_at = current_unix_ts().max(started_at);

    // Ensure every test_id referenced by test_outcomes exists in `tests`.
    let mut tests_to_upsert = discovered;
    let mut known_ids: HashSet<String> = tests_to_upsert
        .iter()
        .map(|t| t.test_id.clone())
        .collect();
    for outcome in &result.outcomes {
        if known_ids.insert(outcome.test_id.clone()) {
            let file_path = outcome
                .test_id
                .split("::")
                .next()
                .map(normalize_posix_path)
                .unwrap_or_else(|| outcome.test_id.clone());
            tests_to_upsert.push(TestCase {
                test_id: outcome.test_id.clone(),
                file_path,
            });
        }
    }

    let selected_by_id: HashMap<&str, &SelectedTest> = selected_plan
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|s| (s.test_id.as_str(), s))
        .collect();

    let run_record = RunRecord {
        run_id: run_id.clone(),
        commit_sha: diff.base_commit.clone(),
        diff_fingerprint: diff.diff_fingerprint.clone(),
        runner: runner.name().to_string(),
        mode,
        budget_ms,
        started_at,
        finished_at: Some(finished_at),
        total_duration_ms: Some(result.total_duration_ms as i64),
        exit_code: Some(result.exit_code),
    };

    let outcome_records: Vec<TestOutcomeRecord> = result
        .outcomes
        .iter()
        .map(|o| {
            let (predicted_prob, selection_reason, sampling_propensity) =
                if let Some(sel) = selected_by_id.get(o.test_id.as_str()) {
                    (
                        Some(sel.predicted_prob),
                        sel.selection_reason,
                        sel.sampling_propensity,
                    )
                } else {
                    (None, SelectionReason::Full, 1.0)
                };

            TestOutcomeRecord {
                run_id: run_id.clone(),
                test_id: o.test_id.clone(),
                status: o.status,
                duration_ms: o.duration_ms,
                predicted_prob,
                selection_reason,
                sampling_propensity,
                failure_excerpt: o.failure_message.clone(),
            }
        })
        .collect();

    // Invariant 6: Persist outcomes BEFORE emitting report and exiting with runner status.
    store.persist_run_with_outcomes(
        &run_record,
        &diff.files,
        &tests_to_upsert,
        &outcome_records,
        finished_at,
    )?;

    emit_run_report(&run_id, mode, model_mode, &result, args.json)?;
    Ok(result.exit_code)
}

fn emit_dry_run_report(
    mode_label: &str,
    model_mode: &str,
    diff_files_count: usize,
    discovered: &[TestCase],
    selected_plan: Option<&[SelectedTest]>,
    as_json: bool,
) -> Result<()> {
    let items: Vec<DryRunSelectedItem> = match selected_plan {
        Some(plan) => plan
            .iter()
            .map(|s| DryRunSelectedItem {
                test_id: s.test_id.clone(),
                predicted_fail_prob: Some(s.predicted_prob),
                est_duration_ms: Some(s.est_duration_ms),
                reason: s.selection_reason,
                sampling_propensity: s.sampling_propensity,
            })
            .collect(),
        None => discovered
            .iter()
            .map(|tc| DryRunSelectedItem {
                test_id: tc.test_id.clone(),
                predicted_fail_prob: None,
                est_duration_ms: None,
                reason: SelectionReason::Full,
                sampling_propensity: 1.0,
            })
            .collect(),
    };

    let estimated_duration_ms: u64 = items.iter().filter_map(|i| i.est_duration_ms).sum();

    if as_json {
        let report = DryRunJsonReport {
            mode: mode_label.to_string(),
            model_mode: model_mode.to_string(),
            dry_run: true,
            diff_files_count,
            total_discovered_tests: discovered.len(),
            selected_count: items.len(),
            estimated_duration_ms,
            selected: items,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Dry run (mode={mode_label}, model={model_mode}): selected {}/{} test(s), diff_files={diff_files_count}, est_duration={} ms",
            items.len(),
            discovered.len(),
            estimated_duration_ms
        );
        for item in &items {
            match (item.predicted_fail_prob, item.est_duration_ms) {
                (Some(p), Some(ms)) => println!(
                    "  {}  prob={:.3}  est={}ms  reason={}  propensity={:.3}",
                    item.test_id, p, ms, item.reason, item.sampling_propensity
                ),
                _ => println!("  {}  reason={}", item.test_id, item.reason),
            }
        }
    }
    Ok(())
}

fn emit_run_report(
    run_id: &str,
    mode: RunMode,
    model_mode: &str,
    result: &RunnerResult,
    as_json: bool,
) -> Result<()> {
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut errored = 0usize;
    let mut skipped = 0usize;
    let mut failures = Vec::new();

    for outcome in &result.outcomes {
        match outcome.status {
            TestStatus::Pass => passed += 1,
            TestStatus::Fail => {
                failed += 1;
                failures.push(FailureSummary {
                    test_id: outcome.test_id.clone(),
                    duration_ms: outcome.duration_ms,
                    failure_message: outcome.failure_message.clone(),
                });
            }
            TestStatus::Error => {
                errored += 1;
                failures.push(FailureSummary {
                    test_id: outcome.test_id.clone(),
                    duration_ms: outcome.duration_ms,
                    failure_message: outcome.failure_message.clone(),
                });
            }
            TestStatus::Skip => skipped += 1,
        }
    }

    if as_json {
        let report = RunJsonReport {
            run_id: run_id.to_string(),
            mode,
            model_mode: model_mode.to_string(),
            exit_code: result.exit_code,
            total_duration_ms: result.total_duration_ms,
            passed,
            failed,
            errored,
            skipped,
            total: result.outcomes.len(),
            failures,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Run {run_id} (mode={mode}, model={model_mode}): {passed} passed, {failed} failed, {errored} errored, {skipped} skipped in {} ms (exit={})",
            result.total_duration_ms, result.exit_code
        );
        for f in &failures {
            println!("  FAIL {} ({} ms)", f.test_id, f.duration_ms);
            if let Some(msg) = &f.failure_message
                && let Some(first_line) = msg.lines().next()
            {
                println!("       {first_line}");
            }
        }
    }
    Ok(())
}

fn current_unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
