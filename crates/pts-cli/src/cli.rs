use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use pts_collectors::{
    ImportGraph, ImportGraphBuilder, PythonImportGraphBuilder, TempGitWorktree,
    extract_working_diff, is_git_repo, mine_commit_history, mine_commit_history_from_ref,
};
use pts_core::{
    Config, DiffSummary, HistoricalRun, HistorySnapshot, PtsError, RunMode, RunnerResult,
    SelectionReason, TestCase, TestRunner, TestStatus,
};
use pts_eval::{WalkForwardConfig, walk_forward_eval};
use pts_ml::{
    CandidateTest, NUM_FEATURES, SelectedTest, SelectionStrategy, TrainedModel, extract_features,
    score_heuristic, select_tests, should_use_heuristic, train_gbdt_model,
};
use pts_runners::{PytestRunner, normalize_posix_path};
use pts_store::{ModelRecord, RunRecord, Store, TestOutcomeRecord};
use serde::Serialize;
use tracing::{info, warn};
use uuid::Uuid;

use crate::mcp;

const DEFAULT_UNSEEN_STALENESS_DAYS: f64 = 30.0;
const MODEL_ARTIFACT_REL_PATH: &str = ".pts/model.json";

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
    /// Train the GBDT + Platt calibration model from .pts/store.db.
    Train,
    /// Explain the ranking features and score for a specific test (Milestone 4).
    Explain { test_id: String },
    /// Run chronological walk-forward evaluation against baselines.
    Eval(EvalArgs),
    /// Replay historical git commits in an isolated worktree to populate .pts/store.db.
    Backfill(BackfillArgs),
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

#[derive(Debug, Args)]
pub struct EvalArgs {
    /// Optional path to write structured JSON evaluation report.
    #[arg(long)]
    pub output_json: Option<PathBuf>,

    /// Comma-separated budget fractions of full-suite duration (e.g. 0.1,0.2,0.3).
    #[arg(long, default_value = "0.1,0.2,0.3")]
    pub budgets: String,

    /// Minimum number of warmup runs prior to walk-forward evaluation.
    #[arg(long, default_value_t = 20)]
    pub min_train_runs: usize,

    /// Retrain cadence (number of evaluated runs between GBDT retrains).
    #[arg(long, default_value_t = 5)]
    pub retrain_every: usize,
}

#[derive(Debug, Args)]
pub struct BackfillArgs {
    /// Number of historical commits to replay into .pts/store.db.
    #[arg(long, value_name = "N")]
    pub commits: usize,

    /// Starting git ref to walk backwards from (defaults to HEAD).
    #[arg(long, default_value = "HEAD")]
    pub r#ref: String,
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
        Command::Train => {
            cmd_train(repo_root)?;
            Ok(0)
        }
        Command::Explain { .. } => bail!("`pts explain` is planned for Milestone 4"),
        Command::Eval(args) => {
            cmd_eval(repo_root, args)?;
            Ok(0)
        }
        Command::Backfill(args) => {
            cmd_backfill(repo_root, args)?;
            Ok(0)
        }
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

fn cmd_train(repo_root: &Path) -> Result<()> {
    let db_path = repo_root.join(".pts").join("store.db");
    if !db_path.is_file() {
        return Err(PtsError::NotInitialized.into());
    }

    let store = Store::open(&db_path)?;
    store.migrate()?;

    let mut runs = store.load_historical_runs()?;
    let commits = store.load_commits()?;
    if runs.is_empty() {
        bail!("no historical test runs found in {}; run `pts run --all` or `pts backfill` first", db_path.display());
    }

    let graph = PythonImportGraphBuilder::new().build(repo_root)?;
    populate_ast_hops(&mut runs, &graph);

    let trained_at = current_unix_ts();
    let result = train_gbdt_model(&runs, &commits, trained_at)
        .context("failed to train GBDT + Platt model")?;

    let model_path = repo_root.join(MODEL_ARTIFACT_REL_PATH);
    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let artifact_json = result.model.to_json()?;
    std::fs::write(&model_path, &artifact_json)
        .with_context(|| format!("failed to write model artifact to {}", model_path.display()))?;

    let model_id = Uuid::new_v4().to_string();
    let artifact = &result.model.artifact;
    let model_record = ModelRecord {
        model_id: model_id.clone(),
        trained_at: artifact.trained_at,
        max_train_ts: artifact.max_train_ts,
        train_rows: artifact.train_rows,
        pos_rows: artifact.pos_rows,
        brier_score: artifact.brier_score,
        calibration_a: artifact.platt_a,
        calibration_b: artifact.platt_b,
        artifact_path: MODEL_ARTIFACT_REL_PATH.to_string(),
    };
    store.insert_model(&model_record)?;

    info!(
        model_id = %model_id,
        train_rows = artifact.train_rows,
        cal_rows = result.cal_rows,
        brier_score = artifact.brier_score,
        "trained GBDT + Platt model"
    );

    println!("Trained PTS model ({model_id}) -> {}", model_path.display());
    println!(
        "  Train: {} rows ({} positives across {} run(s))",
        artifact.train_rows, artifact.pos_rows, result.train_run_count
    );
    println!(
        "  Cal:   {} rows ({} positives across {} run(s))",
        result.cal_rows, result.cal_pos_rows, result.cal_run_count
    );
    println!(
        "  Brier: {:.6} (uncalibrated: {:.6})",
        result.calibrated_brier, result.uncalibrated_brier
    );
    println!(
        "  Platt: a={:.6}, b={:.6}",
        artifact.platt_a, artifact.platt_b
    );

    Ok(())
}

fn cmd_eval(repo_root: &Path, args: EvalArgs) -> Result<()> {
    let db_path = repo_root.join(".pts").join("store.db");
    if !db_path.is_file() {
        return Err(PtsError::NotInitialized.into());
    }

    let budget_fractions = parse_budgets(&args.budgets)?;
    let config = Config::load(repo_root)?;

    let store = Store::open(&db_path)?;
    store.migrate()?;

    let mut runs = store.load_historical_runs()?;
    let commits = store.load_commits()?;
    let graph = PythonImportGraphBuilder::new().build(repo_root)?;
    populate_ast_hops(&mut runs, &graph);

    let wf_config = WalkForwardConfig {
        min_train_runs: args.min_train_runs,
        retrain_every: args.retrain_every,
        budget_fractions,
        selector_config: config,
    };

    let report = walk_forward_eval(&runs, &commits, &wf_config)
        .context("walk-forward evaluation failed")?;

    print!("{}", report.render_markdown());

    if let Some(output_path) = args.output_json {
        let resolved = if output_path.is_absolute() {
            output_path
        } else {
            repo_root.join(output_path)
        };
        if let Some(parent) = resolved.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let json = report.to_json_pretty()?;
        std::fs::write(&resolved, json)
            .with_context(|| format!("failed to write JSON report to {}", resolved.display()))?;
        info!(path = %resolved.display(), "wrote walk-forward evaluation JSON report");
    }

    Ok(())
}

fn cmd_backfill(repo_root: &Path, args: BackfillArgs) -> Result<()> {
    if args.commits == 0 {
        bail!("--commits must be at least 1");
    }

    let db_path = repo_root.join(".pts").join("store.db");
    if !db_path.is_file() {
        return Err(PtsError::NotInitialized.into());
    }
    if !is_git_repo(repo_root) {
        bail!("`pts backfill` requires a git repository");
    }

    let config = Config::load(repo_root)?;
    let runner = PytestRunner::from_config(&config);
    let mut store = Store::open(&db_path)?;
    store.migrate()?;

    let mut commits = mine_commit_history_from_ref(repo_root, &args.r#ref, args.commits)?;
    if commits.is_empty() {
        println!("No commits found for ref `{}`", args.r#ref);
        return Ok(());
    }

    store.upsert_commits(&commits)?;
    // Process oldest-to-newest so history replays chronologically.
    commits.reverse();

    let worktree = TempGitWorktree::create(repo_root, &commits[0].commit_sha)
        .context("failed to create temporary git worktree for backfill")?;

    let mut persisted = 0usize;
    let mut skipped = 0usize;

    for (idx, commit) in commits.iter().enumerate() {
        if idx > 0
            && let Err(err) = worktree.checkout(&commit.commit_sha)
        {
            warn!(
                commit = %commit.commit_sha,
                error = %err,
                "backfill skipping commit: checkout failed"
            );
            skipped += 1;
            continue;
        }

        let discovered = match runner.discover(worktree.path()) {
            Ok(cases) => cases,
            Err(err) => {
                warn!(
                    commit = %commit.commit_sha,
                    error = %err,
                    "backfill skipping commit: pytest discovery failed"
                );
                skipped += 1;
                continue;
            }
        };

        let result = match runner.execute(worktree.path(), None) {
            Ok(res) => res,
            Err(err) => {
                warn!(
                    commit = %commit.commit_sha,
                    error = %err,
                    "backfill skipping commit: pytest execution failed"
                );
                skipped += 1;
                continue;
            }
        };

        let diff = DiffSummary::from_files(Some(commit.commit_sha.clone()), commit.files.clone());
        let run_id = Uuid::new_v4().to_string();
        let started_at = commit.commit_ts;
        let finished_at = started_at.saturating_add((result.total_duration_ms / 1000) as i64);

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
            commit_sha: Some(commit.commit_sha.clone()),
            diff_fingerprint: diff.diff_fingerprint,
            runner: runner.name().to_string(),
            mode: RunMode::Backfill,
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

        store.persist_run_with_outcomes(
            &run_record,
            &diff.files,
            &tests_to_upsert,
            &outcome_records,
            started_at,
        )?;
        persisted += 1;
    }

    drop(worktree);

    println!(
        "Backfilled {persisted}/{} commit(s) from `{}` ({skipped} skipped)",
        commits.len(),
        args.r#ref
    );
    Ok(())
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

        let loaded_model = try_load_model(repo_root);
        let use_gbdt = !should_use_heuristic(&history, &config, loaded_model.is_some());

        let graph = PythonImportGraphBuilder::new().build(repo_root)?;
        let modified_paths: Vec<&str> = diff.files.iter().map(|f| f.path.as_str()).collect();

        let feature_rows: Vec<[f64; NUM_FEATURES]> = discovered
            .iter()
            .map(|tc| {
                let ast_min_hop = graph.min_hop_to_any(&tc.file_path, &modified_paths);
                extract_features(tc, &diff, &history, ast_min_hop)
            })
            .collect();

        let (probs, active_model_mode, primary_reason) = if use_gbdt
            && let Some(ref model) = loaded_model
        {
            (
                model.predict_probs(&feature_rows),
                "gbdt",
                SelectionReason::Ranked,
            )
        } else {
            (
                feature_rows.iter().map(score_heuristic).collect(),
                "heuristic",
                SelectionReason::Heuristic,
            )
        };

        let candidates: Vec<CandidateTest> = discovered
            .iter()
            .zip(probs)
            .map(|(tc, prob)| {
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
            primary_reason,
        );
        (
            run_mode,
            label,
            active_model_mode,
            budget_ms_opt,
            Some(selected),
        )
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

fn try_load_model(repo_root: &Path) -> Option<TrainedModel> {
    let model_path = repo_root.join(MODEL_ARTIFACT_REL_PATH);
    if !model_path.is_file() {
        return None;
    }
    let raw = match std::fs::read_to_string(&model_path) {
        Ok(content) => content,
        Err(err) => {
            warn!(
                path = %model_path.display(),
                error = %err,
                "failed to read model artifact; falling back to heuristic"
            );
            return None;
        }
    };
    match TrainedModel::from_json(&raw) {
        Ok(model) => Some(model),
        Err(err) => {
            warn!(
                path = %model_path.display(),
                error = %err,
                "invalid model artifact; falling back to heuristic"
            );
            None
        }
    }
}

fn populate_ast_hops(runs: &mut [HistoricalRun], graph: &ImportGraph) {
    for run in runs {
        let modified_paths: Vec<&str> = run.diff_files.iter().map(|f| f.path.as_str()).collect();
        for outcome in &mut run.outcomes {
            outcome.ast_min_hop = graph.min_hop_to_any(&outcome.file_path, &modified_paths);
        }
    }
}

fn parse_budgets(raw: &str) -> Result<Vec<f64>> {
    let mut budgets = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: f64 = trimmed
            .parse()
            .with_context(|| format!("invalid budget fraction '{trimmed}'"))?;
        if !val.is_finite() || val <= 0.0 || val > 1.0 {
            bail!("budget fraction must be in (0.0, 1.0], got {val}");
        }
        budgets.push(val);
    }
    if budgets.is_empty() {
        bail!("--budgets must specify at least one budget fraction in (0.0, 1.0]");
    }
    Ok(budgets)
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
