use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use pts_core::{
    Config, DiffSummary, PtsError, RunMode, SelectionReason, TestCase, TestRunner, TestStatus,
};
use pts_runners::{PytestRunner, normalize_posix_path};
use pts_store::{RunRecord, Store, TestOutcomeRecord};
use serde::Serialize;
use tracing::{info, warn};
use uuid::Uuid;

use crate::mcp;

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
    /// Initialize .pts/ state directory and SQLite store in the current repository.
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
    /// Maximum number of git commits to mine during initialization (Milestone 2).
    #[arg(long, default_value_t = 500)]
    pub history_limit: usize,
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

    /// Time budget in seconds (Milestone 2).
    #[arg(long, value_name = "SECS")]
    pub budget: Option<u64>,

    /// Maximum number of tests to select (Milestone 2).
    #[arg(long, value_name = "N")]
    pub top_k: Option<usize>,

    /// Target failure recall in (0, 1) (Milestone 2).
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
struct RunJsonReport {
    run_id: String,
    mode: RunMode,
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
    let state_dir = repo_root.join(".pts");
    std::fs::create_dir_all(&state_dir)
        .with_context(|| format!("failed to create {}", state_dir.display()))?;

    let db_path = state_dir.join("store.db");
    let store = Store::open(&db_path)?;
    store.migrate()?;

    check_gitignore_warning(repo_root);
    info!(
        db = %db_path.display(),
        history_limit = args.history_limit,
        "initialized PTS store"
    );
    println!("Initialized PTS store at {}", db_path.display());
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
    if !args.all {
        bail!(
            "selective modes (--budget, --top-k, --target-recall) are planned for Milestone 2; use `pts run --all` in Milestone 1"
        );
    }

    let config = Config::load(repo_root)?;
    let runner = PytestRunner::from_config(&config);
    let discovered = runner.discover(repo_root)?;

    if args.dry_run {
        if args.json {
            let payload = serde_json::json!({
                "mode": "all",
                "dry_run": true,
                "selected_count": discovered.len(),
                "selected": discovered.iter().map(|t| &t.test_id).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            println!("Dry run (--all): {} test(s) discovered", discovered.len());
            for tc in &discovered {
                println!("  {}", tc.test_id);
            }
        }
        return Ok(0);
    }

    let db_path = repo_root.join(".pts").join("store.db");
    if !db_path.is_file() {
        return Err(PtsError::NotInitialized.into());
    }
    let mut store = Store::open(&db_path)?;
    store.migrate()?;

    let run_id = Uuid::new_v4().to_string();
    let started_at = current_unix_ts();
    let result = runner.execute(repo_root, None)?;
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

    let run_record = RunRecord {
        run_id: run_id.clone(),
        commit_sha: None,
        diff_fingerprint: DiffSummary::compute_fingerprint(&[]),
        runner: runner.name().to_string(),
        mode: RunMode::All,
        budget_ms: None,
        started_at,
        finished_at: Some(finished_at),
        total_duration_ms: Some(result.total_duration_ms as i64),
        exit_code: Some(result.exit_code),
    };

    let outcome_records: Vec<TestOutcomeRecord> = result
        .outcomes
        .iter()
        .map(|o| TestOutcomeRecord {
            run_id: run_id.clone(),
            test_id: o.test_id.clone(),
            status: o.status,
            duration_ms: o.duration_ms,
            predicted_prob: None,
            selection_reason: SelectionReason::Full,
            sampling_propensity: 1.0,
            failure_excerpt: o.failure_message.clone(),
        })
        .collect();

    // Invariant 6: Persist outcomes BEFORE emitting report and exiting with runner status.
    store.persist_run_with_outcomes(
        &run_record,
        &[],
        &tests_to_upsert,
        &outcome_records,
        finished_at,
    )?;

    emit_run_report(&run_id, &result, args.json)?;
    Ok(result.exit_code)
}

fn emit_run_report(
    run_id: &str,
    result: &pts_core::RunnerResult,
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
            mode: RunMode::All,
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
            "Run {run_id} (mode=all): {passed} passed, {failed} failed, {errored} errored, {skipped} skipped in {} ms (exit={})",
            result.total_duration_ms, result.exit_code
        );
        for f in &failures {
            println!("  FAIL {} ({} ms)", f.test_id, f.duration_ms);
            if let Some(msg) = &f.failure_message {
                if let Some(first_line) = msg.lines().next() {
                    println!("       {first_line}");
                }
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
