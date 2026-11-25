use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use failcast_collectors::{
    ImportGraph, ImportGraphBuilder, PythonImportGraphBuilder, extract_working_diff, is_git_repo,
};
use failcast_core::{
    Config, DiffFile, DiffSummary, FailcastError, HistoricalRun, HistorySnapshot, RunMode,
    RunnerResult, SelectionReason, TestCase, TestRunner, TestStatus,
};
use failcast_ml::features::{
    IDX_AST_MIN_HOP, IDX_COCHANGE_FREQ_MAX, IDX_COFAIL_RATE_MAX, IDX_DIFF_FILES_COUNT,
    IDX_DIFF_LINES_TOTAL, IDX_DIFF_TOP_DIRS_COUNT, IDX_DIRECT_TEST_MODIFIED, IDX_EWMA_FAIL_10,
    IDX_EWMA_FAIL_50, IDX_FLAKINESS_RATE, IDX_MEDIAN_DURATION_MS, IDX_PATH_JACCARD_MAX,
    IDX_RUNS_SINCE_LAST_FAIL, IDX_SAME_DIR_OVERLAP, IDX_TOTAL_TEST_RUNS,
    IDX_TOUCHED_FILE_CHURN_30D,
};
use failcast_ml::{
    CandidateTest, DEFAULT_MEDIAN_DURATION_MS, FEATURE_NAMES, MIN_EFF_DURATION_MS, NUM_FEATURES,
    SelectedTest, SelectionStrategy, TrainedModel, extract_features, score_heuristic, select_tests,
    should_use_heuristic,
};
use failcast_runners::{PytestRunner, normalize_posix_path};
use failcast_store::{RunRecord, Store, TestOutcomeRecord};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::warn;
use uuid::Uuid;

pub const DEFAULT_UNSEEN_STALENESS_DAYS: f64 = 30.0;
pub const MODEL_ARTIFACT_REL_PATH: &str = ".failcast/model.json";
pub const DISCOVERY_FINGERPRINT_META_KEY: &str = "discovery_fingerprint";

/// Selection strategy requested by CLI commands or MCP tools.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionRequest {
    All,
    Budget {
        budget_seconds: u64,
    },
    TopK {
        top_k: usize,
    },
    TargetRecall {
        target_recall: f64,
        max_budget_seconds: Option<u64>,
    },
}

/// Result of dry-run working-tree test selection, shared across CLI and MCP.
#[derive(Debug, Clone)]
pub struct SelectionPlan {
    pub config: Config,
    pub diff: DiffSummary,
    pub discovered: Vec<TestCase>,
    pub mode: RunMode,
    pub mode_label: &'static str,
    pub model_mode: &'static str,
    pub budget_ms: Option<i64>,
    pub selected_plan: Option<Vec<SelectedTest>>,
}

/// Structured failure summary shared between CLI and MCP run reports.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FailureSummary {
    pub test_id: String,
    pub duration_ms: u64,
    pub failure_message: Option<String>,
}

/// Complete execution result returned by `run_selected_tests`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunExecutionReport {
    pub run_id: String,
    pub mode: RunMode,
    pub model_mode: String,
    pub exit_code: i32,
    pub total_duration_ms: u64,
    pub passed: usize,
    pub failed: usize,
    pub errored: usize,
    pub skipped: usize,
    pub total: usize,
    pub failures: Vec<FailureSummary>,
}

/// All 16 features extracted for a test against the current working-tree diff.
/// Undefined/missing values (`f64::NAN` in the internal feature vector) are serialized as `null`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FeatureBreakdown {
    pub ast_min_hop: Option<f64>,
    pub direct_test_modified: f64,
    pub path_jaccard_max: f64,
    pub same_dir_overlap: f64,
    pub cofail_rate_max: f64,
    pub cochange_freq_max: f64,
    pub ewma_fail_10: f64,
    pub ewma_fail_50: f64,
    pub runs_since_last_fail: Option<f64>,
    pub total_test_runs: f64,
    pub median_duration_ms: Option<f64>,
    pub flakiness_rate: f64,
    pub diff_files_count: f64,
    pub diff_lines_total: f64,
    pub diff_top_dirs_count: f64,
    pub touched_file_churn_30d: f64,
}

impl FeatureBreakdown {
    pub fn from_array(f: &[f64; NUM_FEATURES]) -> Self {
        Self {
            ast_min_hop: finite_opt(f[IDX_AST_MIN_HOP]),
            direct_test_modified: finite_or_zero(f[IDX_DIRECT_TEST_MODIFIED]),
            path_jaccard_max: finite_or_zero(f[IDX_PATH_JACCARD_MAX]),
            same_dir_overlap: finite_or_zero(f[IDX_SAME_DIR_OVERLAP]),
            cofail_rate_max: finite_or_zero(f[IDX_COFAIL_RATE_MAX]),
            cochange_freq_max: finite_or_zero(f[IDX_COCHANGE_FREQ_MAX]),
            ewma_fail_10: finite_or_zero(f[IDX_EWMA_FAIL_10]),
            ewma_fail_50: finite_or_zero(f[IDX_EWMA_FAIL_50]),
            runs_since_last_fail: finite_opt(f[IDX_RUNS_SINCE_LAST_FAIL]),
            total_test_runs: finite_or_zero(f[IDX_TOTAL_TEST_RUNS]),
            median_duration_ms: finite_opt(f[IDX_MEDIAN_DURATION_MS]),
            flakiness_rate: finite_or_zero(f[IDX_FLAKINESS_RATE]),
            diff_files_count: finite_or_zero(f[IDX_DIFF_FILES_COUNT]),
            diff_lines_total: finite_or_zero(f[IDX_DIFF_LINES_TOTAL]),
            diff_top_dirs_count: finite_or_zero(f[IDX_DIFF_TOP_DIRS_COUNT]),
            touched_file_churn_30d: finite_or_zero(f[IDX_TOUCHED_FILE_CHURN_30D]),
        }
    }

    pub fn as_rows(&self) -> [(&'static str, Option<f64>); NUM_FEATURES] {
        [
            (FEATURE_NAMES[IDX_AST_MIN_HOP], self.ast_min_hop),
            (
                FEATURE_NAMES[IDX_DIRECT_TEST_MODIFIED],
                Some(self.direct_test_modified),
            ),
            (
                FEATURE_NAMES[IDX_PATH_JACCARD_MAX],
                Some(self.path_jaccard_max),
            ),
            (
                FEATURE_NAMES[IDX_SAME_DIR_OVERLAP],
                Some(self.same_dir_overlap),
            ),
            (
                FEATURE_NAMES[IDX_COFAIL_RATE_MAX],
                Some(self.cofail_rate_max),
            ),
            (
                FEATURE_NAMES[IDX_COCHANGE_FREQ_MAX],
                Some(self.cochange_freq_max),
            ),
            (FEATURE_NAMES[IDX_EWMA_FAIL_10], Some(self.ewma_fail_10)),
            (FEATURE_NAMES[IDX_EWMA_FAIL_50], Some(self.ewma_fail_50)),
            (
                FEATURE_NAMES[IDX_RUNS_SINCE_LAST_FAIL],
                self.runs_since_last_fail,
            ),
            (
                FEATURE_NAMES[IDX_TOTAL_TEST_RUNS],
                Some(self.total_test_runs),
            ),
            (
                FEATURE_NAMES[IDX_MEDIAN_DURATION_MS],
                self.median_duration_ms,
            ),
            (FEATURE_NAMES[IDX_FLAKINESS_RATE], Some(self.flakiness_rate)),
            (
                FEATURE_NAMES[IDX_DIFF_FILES_COUNT],
                Some(self.diff_files_count),
            ),
            (
                FEATURE_NAMES[IDX_DIFF_LINES_TOTAL],
                Some(self.diff_lines_total),
            ),
            (
                FEATURE_NAMES[IDX_DIFF_TOP_DIRS_COUNT],
                Some(self.diff_top_dirs_count),
            ),
            (
                FEATURE_NAMES[IDX_TOUCHED_FILE_CHURN_30D],
                Some(self.touched_file_churn_30d),
            ),
        ]
    }
}

/// Concise explanation of an active feature contributing to a test's ranking.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContributingSignal {
    pub feature: String,
    pub value: f64,
    pub description: String,
}

/// Detailed score and 16-feature explanation for a single test.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExplainReport {
    pub test_id: String,
    pub file_path: String,
    pub model_mode: String,
    pub predicted_fail_prob: f64,
    pub est_duration_ms: u64,
    pub yield_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_gbdt_margin: Option<f64>,
    pub diff_files_count: usize,
    pub features: FeatureBreakdown,
    pub top_signals: Vec<ContributingSignal>,
}

/// Selects tests for the current working-tree diff without executing them or writing runs/outcomes
/// to `.failcast/store.db` (discovery metadata cache may be refreshed on fingerprint mismatch).
pub fn select_for_working_tree(
    repo_root: &Path,
    request: SelectionRequest,
    refresh_tests: bool,
) -> Result<SelectionPlan> {
    if let SelectionRequest::TargetRecall { target_recall, .. } = request
        && (!target_recall.is_finite() || target_recall <= 0.0 || target_recall >= 1.0)
    {
        bail!("target_recall must be strictly between 0.0 and 1.0 (got {target_recall})");
    }

    let config = Config::load(repo_root)?;
    let db_path = repo_root.join(".failcast").join("store.db");
    let diff = load_working_diff(repo_root)?;

    let discovered = discover_tests_cached(repo_root, &config, &db_path, refresh_tests)?;
    let now_ts = current_unix_ts();

    let (mode, mode_label, model_mode, budget_ms, selected_plan): (
        RunMode,
        &'static str,
        &'static str,
        Option<i64>,
        Option<Vec<SelectedTest>>,
    ) = if matches!(request, SelectionRequest::All) {
        (RunMode::All, "all", "full", None, None)
    } else {
        let modified_paths: Vec<&str> = diff.files.iter().map(|f| f.path.as_str()).collect();
        let history = load_history_or_default(&db_path, now_ts, &modified_paths)?;
        let loaded_model = try_load_model(repo_root);
        let use_gbdt = !should_use_heuristic(&history, &config, loaded_model.is_some());

        let graph = PythonImportGraphBuilder::new().build(repo_root)?;

        let feature_rows: Vec<[f64; NUM_FEATURES]> = discovered
            .iter()
            .map(|tc| {
                let ast_min_hop = graph.min_hop_to_any(&tc.file_path, &modified_paths);
                extract_features(tc, &diff, &history, ast_min_hop)
            })
            .collect();

        let (probs, active_model_mode, primary_reason) =
            if use_gbdt && let Some(ref model) = loaded_model {
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

        let (strategy, run_mode, label, budget_ms_opt) = match request {
            SelectionRequest::Budget { budget_seconds } => {
                let ms = budget_seconds.saturating_mul(1000);
                (
                    SelectionStrategy::Budget { budget_ms: ms },
                    RunMode::Budget,
                    "budget",
                    Some(ms as i64),
                )
            }
            SelectionRequest::TopK { top_k } => (
                SelectionStrategy::TopK { k: top_k },
                RunMode::TopK,
                "top_k",
                None,
            ),
            SelectionRequest::TargetRecall {
                target_recall,
                max_budget_seconds,
            } => {
                let max_budget_ms = max_budget_seconds.map(|s| s.saturating_mul(1000));
                (
                    SelectionStrategy::TargetRecall {
                        target_recall,
                        max_budget_ms,
                    },
                    RunMode::Budget,
                    "target_recall",
                    max_budget_ms.map(|ms| ms as i64),
                )
            }
            SelectionRequest::All => unreachable!(),
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

    Ok(SelectionPlan {
        config,
        diff,
        discovered,
        mode,
        mode_label,
        model_mode,
        budget_ms,
        selected_plan,
    })
}

/// Selects and executes tests for the current working tree, persisting all outcomes to `.failcast/store.db`
/// before returning.
pub fn run_selected_tests(
    repo_root: &Path,
    request: SelectionRequest,
    refresh_tests: bool,
) -> Result<RunExecutionReport> {
    let db_path = repo_root.join(".failcast").join("store.db");
    if !db_path.is_file() {
        return Err(FailcastError::NotInitialized.into());
    }

    let plan = select_for_working_tree(repo_root, request, refresh_tests)?;
    let runner = PytestRunner::from_config(&plan.config);

    let mut store = Store::open(&db_path)?;
    store.migrate()?;

    let run_id = Uuid::new_v4().to_string();
    let started_at = current_unix_ts();

    let result = match &plan.selected_plan {
        None => runner.execute(repo_root, None)?,
        Some(selected) if selected.is_empty() => RunnerResult {
            exit_code: 0,
            total_duration_ms: 0,
            outcomes: Vec::new(),
        },
        Some(selected) => {
            let selected_ids: Vec<String> = selected.iter().map(|s| s.test_id.clone()).collect();
            runner.execute(repo_root, Some(&selected_ids))?
        }
    };
    let finished_at = current_unix_ts().max(started_at);

    // Ensure every test_id referenced by test_outcomes exists in `tests`.
    let mut tests_to_upsert = plan.discovered;
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

    let selected_by_id: HashMap<&str, &SelectedTest> = plan
        .selected_plan
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|s| (s.test_id.as_str(), s))
        .collect();

    let run_record = RunRecord {
        run_id: run_id.clone(),
        commit_sha: plan.diff.base_commit.clone(),
        diff_fingerprint: plan.diff.diff_fingerprint.clone(),
        runner: runner.name().to_string(),
        mode: plan.mode,
        budget_ms: plan.budget_ms,
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

    // Persist outcomes before returning so failing test runs still record history.
    store.persist_run_with_outcomes(
        &run_record,
        &plan.diff.files,
        &tests_to_upsert,
        &outcome_records,
        finished_at,
    )?;

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

    Ok(RunExecutionReport {
        run_id,
        mode: plan.mode,
        model_mode: plan.model_mode.to_string(),
        exit_code: result.exit_code,
        total_duration_ms: result.total_duration_ms,
        passed,
        failed,
        errored,
        skipped,
        total: result.outcomes.len(),
        failures,
    })
}

/// Extracts all 16 features for `test_id` against the current working-tree diff,
/// scores it with the active model (GBDT+Platt or cold-start heuristic), and returns
/// the full breakdown and top contributing signals.
pub fn explain_test(repo_root: &Path, test_id: &str, refresh_tests: bool) -> Result<ExplainReport> {
    let config = Config::load(repo_root)?;
    let db_path = repo_root.join(".failcast").join("store.db");
    let discovered = discover_tests_cached(repo_root, &config, &db_path, refresh_tests)?;

    let normalized_query = normalize_test_id(test_id);
    let Some(test_case) = discovered
        .iter()
        .find(|tc| tc.test_id == test_id || tc.test_id == normalized_query)
    else {
        bail!(
            "test `{test_id}` was not found among {} discovered test(s)",
            discovered.len()
        );
    };

    let diff = load_working_diff(repo_root)?;
    let now_ts = current_unix_ts();
    let modified_paths: Vec<&str> = diff.files.iter().map(|f| f.path.as_str()).collect();
    let history = load_history_or_default(&db_path, now_ts, &modified_paths)?;
    let loaded_model = try_load_model(repo_root);
    let use_gbdt = !should_use_heuristic(&history, &config, loaded_model.is_some());

    let graph = PythonImportGraphBuilder::new().build(repo_root)?;
    let ast_min_hop = graph.min_hop_to_any(&test_case.file_path, &modified_paths);
    let features = extract_features(test_case, &diff, &history, ast_min_hop);

    let (model_mode, predicted_fail_prob, raw_gbdt_margin) =
        if use_gbdt && let Some(ref model) = loaded_model {
            let margin = model
                .predict_raw_margins(std::slice::from_ref(&features))
                .into_iter()
                .next()
                .unwrap_or(0.0);
            let prob = model.calibrator().predict_prob(margin);
            ("gbdt", prob, Some(margin))
        } else {
            ("heuristic", score_heuristic(&features), None)
        };

    let fallback_ms = history
        .median_repo_test_ms
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_MEDIAN_DURATION_MS);
    let eff_ms = history
        .test_history
        .get(&test_case.test_id)
        .and_then(|h| h.median_duration_ms)
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(fallback_ms)
        .max(MIN_EFF_DURATION_MS);
    let est_duration_ms = eff_ms.round() as u64;
    let yield_score = predicted_fail_prob / eff_ms;

    let breakdown = FeatureBreakdown::from_array(&features);
    let top_signals = compute_top_signals(&features);

    Ok(ExplainReport {
        test_id: test_case.test_id.clone(),
        file_path: test_case.file_path.clone(),
        model_mode: model_mode.to_string(),
        predicted_fail_prob,
        est_duration_ms,
        yield_score,
        raw_gbdt_margin,
        diff_files_count: diff.files.len(),
        features: breakdown,
        top_signals,
    })
}

/// Loads a previously trained model from `.failcast/model.json`, returning `None` if absent or invalid.
pub fn try_load_model(repo_root: &Path) -> Option<TrainedModel> {
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

/// Populates directed AST hop distances on historical run outcomes using the provided import graph.
pub fn populate_ast_hops(runs: &mut [HistoricalRun], graph: &ImportGraph) {
    for run in runs {
        let modified_paths: Vec<&str> = run.diff_files.iter().map(|f| f.path.as_str()).collect();
        for outcome in &mut run.outcomes {
            outcome.ast_min_hop = graph.min_hop_to_any(&outcome.file_path, &modified_paths);
        }
    }
}

/// Returns current UNIX epoch timestamp in seconds.
pub fn current_unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Truncates a string to at most `max_chars` Unicode scalar values.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect()
}

/// Normalizes diff file entries into canonical POSIX paths, combining duplicate paths.
pub fn normalize_diff_files(raw: Vec<DiffFile>) -> Vec<DiffFile> {
    let mut by_path: std::collections::BTreeMap<String, (u32, u32)> =
        std::collections::BTreeMap::new();
    for file in raw {
        let path = normalize_posix_path(&file.path);
        if path.is_empty() {
            continue;
        }
        let entry = by_path.entry(path).or_insert((0, 0));
        entry.0 = entry.0.saturating_add(file.lines_added);
        entry.1 = entry.1.saturating_add(file.lines_deleted);
    }
    by_path
        .into_iter()
        .map(|(path, (lines_added, lines_deleted))| DiffFile {
            path,
            lines_added,
            lines_deleted,
        })
        .collect()
}

/// Computes a fast deterministic SHA-256 fingerprint over sorted repo-relative paths, byte lengths,
/// and nanosecond modification timestamps of Python test and pytest configuration files.
pub fn compute_discovery_fingerprint(repo_root: &Path) -> Result<String> {
    let mut entries: Vec<(String, u64, u128)> = Vec::new();
    collect_discovery_files(repo_root, repo_root, &mut entries)?;
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    for (rel_path, len, mtime_ns) in entries {
        hasher.update(rel_path.as_bytes());
        hasher.update(b"\0");
        hasher.update(len.to_le_bytes());
        hasher.update(mtime_ns.to_le_bytes());
        hasher.update(b"\n");
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    Ok(hex)
}

fn collect_discovery_files(
    repo_root: &Path,
    dir: &Path,
    out: &mut Vec<(String, u64, u128)>,
) -> Result<()> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read directory {}", dir.display()));
        }
    };

    for entry in read_dir {
        let entry = entry?;
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();
        let path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_symlink() {
            continue;
        }

        if file_type.is_dir() {
            if is_skipped_discovery_dir(&name_str) {
                continue;
            }
            collect_discovery_files(repo_root, &path, out)?;
            continue;
        }

        if !file_type.is_file() {
            continue;
        }

        let rel = path.strip_prefix(repo_root).unwrap_or(&path);
        let rel_posix = normalize_posix_path(&rel.to_string_lossy());
        if !is_discovery_relevant_file(&rel_posix, &name_str) {
            continue;
        }

        let meta = entry.metadata()?;
        let len = meta.len();
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        out.push((rel_posix, len, mtime_ns));
    }

    Ok(())
}

fn is_skipped_discovery_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".failcast"
            | "target"
            | "node_modules"
            | "__pycache__"
            | ".pytest_cache"
            | ".tox"
            | ".mypy_cache"
            | ".ruff_cache"
            | ".venv"
            | "venv"
            | "env"
            | "dist"
            | "build"
            | ".eggs"
            | ".hg"
            | ".svn"
    ) || name.ends_with(".egg-info")
}

fn is_discovery_relevant_file(rel_posix: &str, file_name: &str) -> bool {
    if matches!(
        file_name,
        "pyproject.toml" | "pytest.ini" | "setup.cfg" | "tox.ini" | "conftest.py"
    ) {
        return true;
    }

    if !file_name.ends_with(".py") {
        return false;
    }

    file_name.starts_with("test_")
        || file_name.ends_with("_test.py")
        || rel_posix.starts_with("tests/")
        || rel_posix.contains("/tests/")
}

fn discover_tests_cached(
    repo_root: &Path,
    config: &Config,
    db_path: &Path,
    refresh_tests: bool,
) -> Result<Vec<TestCase>> {
    let runner = PytestRunner::from_config(config);
    if !db_path.is_file() {
        return Ok(runner.discover(repo_root)?);
    }

    let fingerprint = compute_discovery_fingerprint(repo_root)?;
    let mut store = Store::open(db_path)?;
    store.migrate()?;

    if !refresh_tests
        && store.get_meta(DISCOVERY_FINGERPRINT_META_KEY)?.as_deref() == Some(fingerprint.as_str())
    {
        let cached = store.load_active_tests()?;
        if !cached.is_empty() {
            return Ok(cached);
        }
    }

    let discovered = runner.discover(repo_root)?;
    store.sync_discovered_tests(&discovered, &fingerprint, current_unix_ts())?;
    Ok(discovered)
}

fn load_working_diff(repo_root: &Path) -> Result<DiffSummary> {
    if is_git_repo(repo_root) {
        extract_working_diff(repo_root).context("failed to extract working-tree git diff")
    } else {
        warn!(
            repo = %repo_root.display(),
            "repository is not a git working tree; using empty diff"
        );
        Ok(DiffSummary::from_files(None, Vec::new()))
    }
}

fn load_history_or_default(
    db_path: &Path,
    now_ts: i64,
    diff_paths: &[&str],
) -> Result<HistorySnapshot> {
    if db_path.is_file() {
        let store = Store::open(db_path)?;
        Ok(store.load_history_snapshot_for_diff(now_ts, diff_paths)?)
    } else {
        Ok(HistorySnapshot::default())
    }
}

fn normalize_test_id(raw: &str) -> String {
    if let Some((file_part, rest)) = raw.split_once("::") {
        format!("{}::{rest}", normalize_posix_path(file_part))
    } else {
        normalize_posix_path(raw)
    }
}

fn finite_opt(v: f64) -> Option<f64> {
    if v.is_finite() { Some(v) } else { None }
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

fn compute_top_signals(f: &[f64; NUM_FEATURES]) -> Vec<ContributingSignal> {
    let mut ranked: Vec<(f64, ContributingSignal)> = Vec::new();

    if f[IDX_DIRECT_TEST_MODIFIED] > 0.0 {
        ranked.push((
            3.0,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_DIRECT_TEST_MODIFIED].to_string(),
                value: f[IDX_DIRECT_TEST_MODIFIED],
                description: "test file itself is modified in the working diff".to_string(),
            },
        ));
    }

    if f[IDX_AST_MIN_HOP].is_finite() {
        let hop = f[IDX_AST_MIN_HOP];
        let strength = 2.5 / (1.0 + hop.max(0.0));
        let desc = if hop == 0.0 {
            "AST hop distance 0 (test file directly touched)".to_string()
        } else {
            format!("test imports modified code within {hop:.0} AST hop(s)")
        };
        ranked.push((
            strength,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_AST_MIN_HOP].to_string(),
                value: hop,
                description: desc,
            },
        ));
    }

    // Prior cofail rate for unseen pairs is (0 + 0.5) / (0 + 5.0) = 0.10.
    if f[IDX_COFAIL_RATE_MAX].is_finite() && f[IDX_COFAIL_RATE_MAX] > 0.10 + 1e-9 {
        let v = f[IDX_COFAIL_RATE_MAX];
        ranked.push((
            2.2 * v,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_COFAIL_RATE_MAX].to_string(),
                value: v,
                description: format!(
                    "elevated historical co-failure rate ({v:.3}) with modified file(s)"
                ),
            },
        ));
    }

    if f[IDX_COCHANGE_FREQ_MAX].is_finite() && f[IDX_COCHANGE_FREQ_MAX] > 0.0 {
        let v = f[IDX_COCHANGE_FREQ_MAX];
        ranked.push((
            1.5 * v,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_COCHANGE_FREQ_MAX].to_string(),
                value: v,
                description: format!(
                    "frequently co-changed in git history ({:.0}% of modified file commits)",
                    v * 100.0
                ),
            },
        ));
    }

    if f[IDX_PATH_JACCARD_MAX].is_finite() && f[IDX_PATH_JACCARD_MAX] > 0.0 {
        let v = f[IDX_PATH_JACCARD_MAX];
        ranked.push((
            1.5 * v,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_PATH_JACCARD_MAX].to_string(),
                value: v,
                description: format!("path token overlap (Jaccard={v:.2}) with modified file(s)"),
            },
        ));
    }

    if f[IDX_EWMA_FAIL_10].is_finite() && f[IDX_EWMA_FAIL_10] > 0.0 {
        let v = f[IDX_EWMA_FAIL_10];
        ranked.push((
            1.4 * v,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_EWMA_FAIL_10].to_string(),
                value: v,
                description: format!("recent failure rate EWMA(alpha=0.2) = {v:.3}"),
            },
        ));
    }

    if f[IDX_EWMA_FAIL_50].is_finite() && f[IDX_EWMA_FAIL_50] > 0.0 {
        let v = f[IDX_EWMA_FAIL_50];
        ranked.push((
            0.9 * v,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_EWMA_FAIL_50].to_string(),
                value: v,
                description: format!("medium-term failure rate EWMA(alpha=0.04) = {v:.3}"),
            },
        ));
    }

    if f[IDX_SAME_DIR_OVERLAP] > 0.0 {
        ranked.push((
            0.6,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_SAME_DIR_OVERLAP].to_string(),
                value: f[IDX_SAME_DIR_OVERLAP],
                description: "shares parent directory with a modified file".to_string(),
            },
        ));
    }

    if f[IDX_FLAKINESS_RATE].is_finite() && f[IDX_FLAKINESS_RATE] > 0.0 {
        let v = f[IDX_FLAKINESS_RATE];
        ranked.push((
            0.5 * v,
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_FLAKINESS_RATE].to_string(),
                value: v,
                description: format!(
                    "historical flakiness on identical diffs ({:.0}%)",
                    v * 100.0
                ),
            },
        ));
    }

    if f[IDX_RUNS_SINCE_LAST_FAIL].is_finite() {
        let v = f[IDX_RUNS_SINCE_LAST_FAIL];
        ranked.push((
            0.4 / (1.0 + v.max(0.0)),
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_RUNS_SINCE_LAST_FAIL].to_string(),
                value: v,
                description: format!("{v:.0} passing run(s) since last observed failure"),
            },
        ));
    }

    if f[IDX_TOUCHED_FILE_CHURN_30D].is_finite() && f[IDX_TOUCHED_FILE_CHURN_30D] > 0.0 {
        let v = f[IDX_TOUCHED_FILE_CHURN_30D];
        ranked.push((
            0.2 * v.ln_1p(),
            ContributingSignal {
                feature: FEATURE_NAMES[IDX_TOUCHED_FILE_CHURN_30D].to_string(),
                value: v,
                description: format!("modified file(s) had {v:.0} commit(s) in the last 30 days"),
            },
        ));
    }

    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.feature.cmp(&b.1.feature))
    });

    ranked.into_iter().take(5).map(|(_, s)| s).collect()
}
