use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use pts_collectors::{
    ImportGraphBuilder, PythonImportGraphBuilder, TempGitWorktree, is_git_repo,
    mine_commit_history, mine_commit_history_from_ref,
};
use pts_core::{
    Config, DiffFile, DiffSummary, PtsError, RunMode, SelectionReason, TestCase, TestRunner,
    TestStatus,
};
use pts_eval::{WalkForwardConfig, walk_forward_eval};
use pts_ml::{SelectedTest, train_gbdt_model};
use pts_runners::{PytestRunner, normalize_posix_path};
use pts_store::{ModelRecord, RunBatchEntry, RunRecord, Store, TestOutcomeRecord};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

use crate::mcp;
use crate::service::{
    ExplainReport, MODEL_ARTIFACT_REL_PATH, RunExecutionReport, SelectionRequest, current_unix_ts,
    explain_test, normalize_diff_files, populate_ast_hops, run_selected_tests,
    select_for_working_tree, truncate_chars,
};

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
    /// Explain the 16 ranking features and score rationale for a specific test.
    Explain(ExplainArgs),
    /// Run chronological walk-forward evaluation against baselines.
    Eval(EvalArgs),
    /// Replay historical git commits in an isolated worktree to populate .pts/store.db.
    Backfill(BackfillArgs),
    /// Ingest a JSONL file of historical test runs into .pts/store.db (mode = 'backfill').
    ImportRuns(ImportRunsArgs),
    /// Start the local MCP server over stdio.
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
pub struct ExplainArgs {
    /// Canonical test identifier (e.g. "tests/test_calc.py::test_add_positive").
    pub test_id: String,

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

#[derive(Debug, Args)]
pub struct ImportRunsArgs {
    /// Path to newline-delimited JSON (JSONL) file containing historical CI/PR test runs.
    pub path: PathBuf,
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

#[derive(Debug, Deserialize)]
struct ImportRunLine {
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    commit_sha: Option<String>,
    started_at: i64,
    #[serde(default)]
    runner: Option<String>,
    #[serde(default)]
    diff_files: Vec<ImportDiffFile>,
    outcomes: Vec<ImportOutcomeLine>,
}

#[derive(Debug, Deserialize)]
struct ImportDiffFile {
    path: String,
    #[serde(default)]
    lines_added: u32,
    #[serde(default)]
    lines_deleted: u32,
}

#[derive(Debug, Deserialize)]
struct ImportOutcomeLine {
    test_id: String,
    #[serde(default)]
    file_path: Option<String>,
    status: TestStatus,
    #[serde(default)]
    duration_ms: u64,
    #[serde(default)]
    failure_excerpt: Option<String>,
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
        Command::Explain(args) => {
            cmd_explain(repo_root, args)?;
            Ok(0)
        }
        Command::Eval(args) => {
            cmd_eval(repo_root, args)?;
            Ok(0)
        }
        Command::Backfill(args) => {
            cmd_backfill(repo_root, args)?;
            Ok(0)
        }
        Command::ImportRuns(args) => {
            cmd_import_runs(repo_root, args)?;
            Ok(0)
        }
        Command::Mcp => {
            mcp::run_mcp(repo_root)?;
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
        bail!(
            "no historical test runs found in {}; run `pts run --all`, `pts backfill`, or `pts import-runs` first",
            db_path.display()
        );
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

fn cmd_explain(repo_root: &Path, args: ExplainArgs) -> Result<()> {
    let report = explain_test(repo_root, &args.test_id)?;
    emit_explain_report(&report, args.json)?;
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

    let report =
        walk_forward_eval(&runs, &commits, &wf_config).context("walk-forward evaluation failed")?;

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
        let mut known_ids: HashSet<String> =
            tests_to_upsert.iter().map(|t| t.test_id.clone()).collect();
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

fn cmd_import_runs(repo_root: &Path, args: ImportRunsArgs) -> Result<()> {
    let db_path = repo_root.join(".pts").join("store.db");
    if !db_path.is_file() {
        return Err(PtsError::NotInitialized.into());
    }

    let config = Config::load(repo_root)?;
    let resolved = if args.path.is_absolute() {
        args.path
    } else {
        repo_root.join(&args.path)
    };

    let file = File::open(&resolved)
        .with_context(|| format!("failed to open JSONL file {}", resolved.display()))?;
    let reader = BufReader::new(file);

    let mut batch: Vec<RunBatchEntry> = Vec::new();
    let mut total_outcomes = 0usize;

    for (idx, line_res) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line_res
            .with_context(|| format!("failed reading line {line_no} of {}", resolved.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parsed: ImportRunLine = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "invalid JSONL run record on line {line_no} of {}",
                resolved.display()
            )
        })?;

        let run_id = parsed
            .run_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let diff_files = normalize_diff_files(
            parsed
                .diff_files
                .into_iter()
                .map(|df| DiffFile {
                    path: df.path,
                    lines_added: df.lines_added,
                    lines_deleted: df.lines_deleted,
                })
                .collect(),
        );
        let diff_fingerprint = DiffSummary::compute_fingerprint(&diff_files);

        let mut tests_to_upsert: Vec<TestCase> = Vec::with_capacity(parsed.outcomes.len());
        let mut seen_test_ids: HashSet<String> = HashSet::with_capacity(parsed.outcomes.len());
        let mut outcome_records: Vec<TestOutcomeRecord> = Vec::with_capacity(parsed.outcomes.len());
        let mut total_duration_ms: u64 = 0;
        let mut has_failure = false;

        for o in parsed.outcomes {
            total_duration_ms = total_duration_ms.saturating_add(o.duration_ms);
            if matches!(o.status, TestStatus::Fail | TestStatus::Error) {
                has_failure = true;
            }

            let file_path = o
                .file_path
                .as_deref()
                .filter(|p| !p.trim().is_empty())
                .map(normalize_posix_path)
                .unwrap_or_else(|| {
                    o.test_id
                        .split("::")
                        .next()
                        .map(normalize_posix_path)
                        .unwrap_or_else(|| o.test_id.clone())
                });

            if seen_test_ids.insert(o.test_id.clone()) {
                tests_to_upsert.push(TestCase {
                    test_id: o.test_id.clone(),
                    file_path,
                });
            }

            let failure_excerpt = o
                .failure_excerpt
                .map(|msg| truncate_chars(&msg, config.max_traceback_chars));

            outcome_records.push(TestOutcomeRecord {
                run_id: run_id.clone(),
                test_id: o.test_id,
                status: o.status,
                duration_ms: o.duration_ms,
                predicted_prob: None,
                selection_reason: SelectionReason::Full,
                sampling_propensity: 1.0,
                failure_excerpt,
            });
        }

        total_outcomes += outcome_records.len();
        let started_at = parsed.started_at;
        let finished_at = started_at.saturating_add((total_duration_ms / 1000) as i64);

        let run_record = RunRecord {
            run_id,
            commit_sha: parsed.commit_sha,
            diff_fingerprint,
            runner: parsed.runner.unwrap_or_else(|| config.runner.clone()),
            mode: RunMode::Backfill,
            budget_ms: None,
            started_at,
            finished_at: Some(finished_at),
            total_duration_ms: Some(total_duration_ms as i64),
            exit_code: Some(if has_failure { 1 } else { 0 }),
        };

        batch.push(RunBatchEntry {
            run: run_record,
            diff_files,
            tests: tests_to_upsert,
            outcomes: outcome_records,
            seen_ts: started_at,
        });
    }

    let mut store = Store::open(&db_path)?;
    store.migrate()?;
    let imported = store.persist_runs_batch(&batch)?;

    info!(
        path = %resolved.display(),
        runs = imported,
        outcomes = total_outcomes,
        "imported historical runs from JSONL"
    );
    println!(
        "Imported {imported} run(s) ({total_outcomes} test outcome(s)) from {} into {}",
        resolved.display(),
        db_path.display()
    );
    Ok(())
}

fn cmd_run(repo_root: &Path, args: RunArgs) -> Result<i32> {
    let request = if args.all {
        SelectionRequest::All
    } else if let Some(budget_seconds) = args.budget {
        SelectionRequest::Budget { budget_seconds }
    } else if let Some(top_k) = args.top_k {
        SelectionRequest::TopK { top_k }
    } else if let Some(target_recall) = args.target_recall {
        SelectionRequest::TargetRecall {
            target_recall,
            max_budget_seconds: None,
        }
    } else {
        unreachable!("clap ArgGroup guarantees one selection mode is set");
    };

    if args.dry_run {
        let plan = select_for_working_tree(repo_root, request)?;
        emit_dry_run_report(
            plan.mode_label,
            plan.model_mode,
            plan.diff.files.len(),
            &plan.discovered,
            plan.selected_plan.as_deref(),
            args.json,
        )?;
        return Ok(0);
    }

    let report = run_selected_tests(repo_root, request)?;
    emit_run_report(&report, args.json)?;
    Ok(report.exit_code)
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

fn emit_explain_report(report: &ExplainReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("Test:         {}", report.test_id);
    println!("File:         {}", report.file_path);
    println!("Model Mode:   {}", report.model_mode);
    println!("Fail Prob:    {:.4}", report.predicted_fail_prob);
    if let Some(margin) = report.raw_gbdt_margin {
        println!("Raw Margin:   {margin:.6}");
    }
    println!("Est Duration: {} ms", report.est_duration_ms);
    println!("Yield Score:  {:.6} prob/ms", report.yield_score);
    println!("Diff Files:   {}", report.diff_files_count);

    println!("\nTop Contributing Signals:");
    if report.top_signals.is_empty() {
        println!("  (no active diff/history signals; baseline prior score)");
    } else {
        for signal in &report.top_signals {
            println!(
                "  - {:<24} = {:<10} ({})",
                signal.feature,
                format_feature_value(Some(signal.value)),
                signal.description
            );
        }
    }

    println!("\n16-Feature Breakdown:");
    println!("  {:>2}  {:<26} {:>12}", "#", "Feature", "Value");
    println!("  {:-<2}  {:-<26} {:-<12}", "", "", "");
    for (idx, (name, value)) in report.features.as_rows().iter().enumerate() {
        println!(
            "  {:>2}  {:<26} {:>12}",
            idx,
            name,
            format_feature_value(*value)
        );
    }

    Ok(())
}

fn format_feature_value(val: Option<f64>) -> String {
    match val {
        None => "NaN".to_string(),
        Some(v) if (v - v.round()).abs() < 1e-9 && v.abs() < 1e9 => format!("{v:.1}"),
        Some(v) => format!("{v:.4}"),
    }
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

fn emit_run_report(report: &RunExecutionReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        println!(
            "Run {} (mode={}, model={}): {} passed, {} failed, {} errored, {} skipped in {} ms (exit={})",
            report.run_id,
            report.mode,
            report.model_mode,
            report.passed,
            report.failed,
            report.errored,
            report.skipped,
            report.total_duration_ms,
            report.exit_code
        );
        for f in &report.failures {
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
