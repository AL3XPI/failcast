use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use pts_core::{
    CommitRecord, Config, DiffSummary, HistoricalOutcome, HistoricalRun, PtsError, SelectionReason,
    TestCase, TestStatus,
};
use pts_ml::{
    CandidateTest, HistoryAccumulator, NUM_FEATURES, TrainedModel, extract_features,
    score_heuristic, train_gbdt_model,
};
use serde::{Deserialize, Serialize};

use crate::baselines::{
    RankerKind, score_random_batch, score_recent_failure_rate, score_static_ast, select_for_budget,
};
use crate::metrics::{MetricsAccumulator, StrategyMetrics};

const DEFAULT_UNSEEN_STALENESS_DAYS: f64 = 30.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Minimum number of chronological runs used before evaluation begins.
    pub min_train_runs: usize,
    /// Retrain GBDT+Platt every `retrain_every` evaluated runs (minimum 1).
    pub retrain_every: usize,
    /// Budget fractions of each run's full-suite duration (e.g. `[0.1, 0.2, 0.3]`).
    pub budget_fractions: Vec<f64>,
    /// Selector hyperparameters (explore_fraction, etc.).
    pub selector_config: Config,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            min_train_runs: 20,
            retrain_every: 5,
            budget_fractions: vec![0.1, 0.2, 0.3],
            selector_config: Config::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetEvalReport {
    pub budget_fraction: f64,
    pub strategies: Vec<StrategyMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalkForwardReport {
    pub total_runs: usize,
    pub evaluated_runs: usize,
    pub failing_runs: usize,
    pub min_train_runs: usize,
    pub retrain_every: usize,
    pub gbdt_steps: usize,
    pub zero_leakage_verified: bool,
    pub budgets: Vec<BudgetEvalReport>,
}

impl WalkForwardReport {
    /// Renders a clean GitHub-flavored Markdown comparison table across all evaluated budgets.
    pub fn render_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# PTS Chronological Walk-Forward Evaluation\n"
        );
        let _ = writeln!(
            out,
            "- **Total runs:** {} | **Evaluated runs:** {} | **Failing runs:** {}",
            self.total_runs, self.evaluated_runs, self.failing_runs
        );
        let _ = writeln!(
            out,
            "- **Warmup runs:** {} | **Retrain cadence:** every {} run(s) | **GBDT steps:** {} | **Zero leakage verified:** {}\n",
            self.min_train_runs,
            self.retrain_every,
            self.gbdt_steps,
            self.zero_leakage_verified
        );

        for budget in &self.budgets {
            let _ = writeln!(
                out,
                "## Budget: {:.0}% of Full Suite Duration\n",
                budget.budget_fraction * 100.0
            );
            let _ = writeln!(
                out,
                "| Strategy | Commit FDR | Failure Recall | Time Saved | Brier Score | PR-AUC |"
            );
            let _ = writeln!(
                out,
                "| :--- | ---: | ---: | ---: | ---: | ---: |"
            );
            for m in &budget.strategies {
                let _ = writeln!(
                    out,
                    "| {} | {:.1}% | {:.1}% | {:.1}% | {:.4} | {:.4} |",
                    m.strategy,
                    m.commit_fdr * 100.0,
                    m.individual_recall * 100.0,
                    m.wall_clock_time_saved * 100.0,
                    m.brier_score,
                    m.pr_auc,
                );
            }
            let _ = writeln!(out);
        }

        out
    }

    pub fn to_json_pretty(&self) -> Result<String, PtsError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PtsError::Parse(format!("failed to serialize walk-forward report: {e}")))
    }

    /// Looks up metrics for a specific budget fraction (within 1e-6) and strategy display name.
    pub fn find_metrics(&self, budget_fraction: f64, strategy: &str) -> Option<&StrategyMetrics> {
        self.budgets
            .iter()
            .find(|b| (b.budget_fraction - budget_fraction).abs() < 1e-6)
            .and_then(|b| b.strategies.iter().find(|s| s.strategy == strategy))
    }
}

/// Runs a strict chronological walk-forward evaluation comparing `PTS (GBDT+Platt)` against
/// `Heuristic`, `StaticAst`, `RecentFailureRate`, and `Random` baselines.
///
/// Asserts `max_train_ts < eval_run.started_at` on every single evaluation step.
pub fn walk_forward_eval(
    runs: &[HistoricalRun],
    commits: &[CommitRecord],
    config: &WalkForwardConfig,
) -> Result<WalkForwardReport, PtsError> {
    if config.budget_fractions.is_empty() {
        return Err(PtsError::InvalidInput(
            "budget_fractions must contain at least one value in (0.0, 1.0]".to_string(),
        ));
    }
    for &frac in &config.budget_fractions {
        if !frac.is_finite() || frac <= 0.0 || frac > 1.0 {
            return Err(PtsError::InvalidInput(format!(
                "invalid budget fraction {frac}: must be in (0.0, 1.0]"
            )));
        }
    }

    let mut sorted_runs: Vec<HistoricalRun> = runs.to_vec();
    sorted_runs.sort_by(|a, b| {
        a.started_at
            .cmp(&b.started_at)
            .then_with(|| a.run_id.cmp(&b.run_id))
    });

    if sorted_runs.len() <= config.min_train_runs || config.min_train_runs < 2 {
        return Err(PtsError::InvalidInput(format!(
            "walk-forward evaluation requires more than min_train_runs ({}) historical runs (got {})",
            config.min_train_runs,
            sorted_runs.len()
        )));
    }

    // Advance start index past any timestamp ties so all warmup runs are strictly `< eval_ts`.
    let mut start_idx = config.min_train_runs;
    while start_idx < sorted_runs.len()
        && sorted_runs[start_idx - 1].started_at == sorted_runs[start_idx].started_at
    {
        start_idx += 1;
    }
    if start_idx >= sorted_runs.len() {
        return Err(PtsError::InvalidInput(
            "cannot separate warmup and evaluation runs with strict monotonic timestamps"
                .to_string(),
        ));
    }

    let retrain_every = config.retrain_every.max(1);
    let mut acc = HistoryAccumulator::new(commits.to_vec());
    for run in &sorted_runs[..start_idx] {
        acc.record_run(run);
    }

    // Accumulators indexed by [budget_idx][strategy_idx].
    let mut metrics_grid: Vec<[MetricsAccumulator; 5]> = config
        .budget_fractions
        .iter()
        .map(|_| std::array::from_fn(|_| MetricsAccumulator::default()))
        .collect();

    let mut current_model: Option<TrainedModel> = None;
    let mut steps_since_train: usize = 0;
    let mut evaluated_runs = 0usize;
    let mut failing_runs = 0usize;
    let mut gbdt_steps = 0usize;
    let mut zero_leakage_verified = true;

    let mut i = start_idx;
    while i < sorted_runs.len() {
        let eval_ts = sorted_runs[i].started_at;
        // Group consecutive runs sharing identical `started_at` so they all evaluate
        // against strictly-prior runs (`started_at < eval_ts`) with zero intra-timestamp leakage.
        let mut group_end = i + 1;
        while group_end < sorted_runs.len() && sorted_runs[group_end].started_at == eval_ts {
            group_end += 1;
        }

        // Strict training slice: ONLY runs with `started_at < eval_ts`.
        let train_slice = &sorted_runs[..i];
        let max_prior_ts = train_slice.iter().map(|r| r.started_at).max().unwrap_or(i64::MIN);
        assert!(
            max_prior_ts < eval_ts,
            "chronological anti-leakage violation: max_prior_ts ({max_prior_ts}) >= eval_ts ({eval_ts})"
        );

        let should_retrain = current_model.is_none() || steps_since_train >= retrain_every;
        if should_retrain {
            match train_gbdt_model(train_slice, commits, eval_ts) {
                Ok(trained) => {
                    assert!(
                        trained.model.artifact.max_train_ts < eval_ts,
                        "trained model max_train_ts ({}) >= eval_ts ({})",
                        trained.model.artifact.max_train_ts,
                        eval_ts
                    );
                    current_model = Some(trained.model);
                    steps_since_train = 0;
                }
                Err(_) => {
                    // Not yet enough positives/negatives in prior history; fall back to heuristic
                    // on this step and retry next step.
                    current_model = None;
                }
            }
        }

        if let Some(ref model) = current_model {
            if model.artifact.max_train_ts >= eval_ts {
                zero_leakage_verified = false;
            }
            assert!(
                model.artifact.max_train_ts < eval_ts,
                "active model max_train_ts ({}) >= eval_ts ({})",
                model.artifact.max_train_ts,
                eval_ts
            );
        }

        // Strictly prior history snapshot for all runs at `eval_ts`.
        let snapshot = acc.snapshot(eval_ts);

        for eval_run in &sorted_runs[i..group_end] {
            let mut non_skip: Vec<&HistoricalOutcome> = eval_run
                .outcomes
                .iter()
                .filter(|o| o.status != TestStatus::Skip)
                .collect();
            non_skip.sort_by(|a, b| a.test_id.cmp(&b.test_id));

            if non_skip.is_empty() {
                continue;
            }

            evaluated_runs += 1;
            steps_since_train += 1;
            if current_model.is_some() {
                gbdt_steps += 1;
            }

            let diff = DiffSummary {
                base_commit: eval_run.commit_sha.clone(),
                diff_fingerprint: eval_run.diff_fingerprint.clone(),
                files: eval_run.diff_files.clone(),
            };

            let mut feature_rows: Vec<[f64; NUM_FEATURES]> = Vec::with_capacity(non_skip.len());
            let mut test_ids: Vec<String> = Vec::with_capacity(non_skip.len());
            let mut labels: Vec<bool> = Vec::with_capacity(non_skip.len());
            let mut actual_durations: HashMap<&str, u64> = HashMap::with_capacity(non_skip.len());
            let mut failing_ids: HashSet<&str> = HashSet::new();
            let mut est_durations: Vec<Option<f64>> = Vec::with_capacity(non_skip.len());
            let mut staleness_days: Vec<f64> = Vec::with_capacity(non_skip.len());

            let mut full_suite_duration_ms = 0u64;
            for outcome in &non_skip {
                let tc = TestCase {
                    test_id: outcome.test_id.clone(),
                    file_path: outcome.file_path.clone(),
                };
                feature_rows.push(extract_features(&tc, &diff, &snapshot, outcome.ast_min_hop));
                test_ids.push(outcome.test_id.clone());

                let is_fail = matches!(outcome.status, TestStatus::Fail | TestStatus::Error);
                labels.push(is_fail);
                if is_fail {
                    failing_ids.insert(outcome.test_id.as_str());
                }
                actual_durations.insert(outcome.test_id.as_str(), outcome.duration_ms);
                full_suite_duration_ms = full_suite_duration_ms.saturating_add(outcome.duration_ms);

                let hist = snapshot.test_history.get(&outcome.test_id);
                est_durations.push(hist.and_then(|h| h.median_duration_ms));
                let days = hist
                    .and_then(|h| h.last_run_ts)
                    .map(|ts| ((eval_run.started_at - ts).max(0) as f64) / 86_400.0)
                    .unwrap_or(DEFAULT_UNSEEN_STALENESS_DAYS);
                staleness_days.push(days);
            }

            let run_total_failures = failing_ids.len();
            if run_total_failures > 0 {
                failing_runs += 1;
            }

            // Score all 5 strategies on this run's non-skip tests.
            let heuristic_probs: Vec<f64> = feature_rows.iter().map(score_heuristic).collect();
            let pts_probs: Vec<f64> = match &current_model {
                Some(model) => model.predict_probs(&feature_rows),
                None => heuristic_probs.clone(),
            };
            let static_ast_probs: Vec<f64> = feature_rows.iter().map(score_static_ast).collect();
            let recent_fail_probs: Vec<f64> = feature_rows
                .iter()
                .map(score_recent_failure_rate)
                .collect();
            let random_probs = score_random_batch(&test_ids, &eval_run.diff_fingerprint);

            let strategy_probs: [&[f64]; 5] = [
                &pts_probs,
                &heuristic_probs,
                &static_ast_probs,
                &recent_fail_probs,
                &random_probs,
            ];
            let primary_reasons: [SelectionReason; 5] = [
                if current_model.is_some() {
                    SelectionReason::Ranked
                } else {
                    SelectionReason::Heuristic
                },
                SelectionReason::Heuristic,
                SelectionReason::Heuristic,
                SelectionReason::Heuristic,
                SelectionReason::Heuristic,
            ];

            for (b_idx, &budget_frac) in config.budget_fractions.iter().enumerate() {
                let budget_ms = ((full_suite_duration_ms as f64) * budget_frac)
                    .round()
                    .max(1.0) as u64;

                for s_idx in 0..RankerKind::ALL.len() {
                    let probs = strategy_probs[s_idx];
                    let candidates: Vec<CandidateTest> = (0..test_ids.len())
                        .map(|t_idx| CandidateTest {
                            test_id: test_ids[t_idx].clone(),
                            prob: probs[t_idx],
                            est_duration_ms: est_durations[t_idx],
                            days_since_last_run: staleness_days[t_idx],
                        })
                        .collect();

                    let selected = select_for_budget(
                        &candidates,
                        budget_ms,
                        &eval_run.diff_fingerprint,
                        snapshot.median_repo_test_ms,
                        &config.selector_config,
                        primary_reasons[s_idx],
                    );

                    let mut caught_failures = 0usize;
                    let mut selected_duration_ms = 0u64;
                    for sel in &selected {
                        if failing_ids.contains(sel.test_id.as_str()) {
                            caught_failures += 1;
                        }
                        if let Some(&dur) = actual_durations.get(sel.test_id.as_str()) {
                            selected_duration_ms = selected_duration_ms.saturating_add(dur);
                        }
                    }

                    let pred_pairs: Vec<(f64, bool)> = probs
                        .iter()
                        .copied()
                        .zip(labels.iter().copied())
                        .collect();

                    metrics_grid[b_idx][s_idx].record_run(
                        run_total_failures,
                        caught_failures,
                        selected_duration_ms,
                        full_suite_duration_ms,
                        &pred_pairs,
                    );
                }
            }
        }

        // Only incorporate `eval_run` outcomes AFTER evaluating all runs at `eval_ts`.
        for eval_run in &sorted_runs[i..group_end] {
            acc.record_run(eval_run);
        }

        i = group_end;
    }

    let budgets: Vec<BudgetEvalReport> = config
        .budget_fractions
        .iter()
        .enumerate()
        .map(|(b_idx, &budget_fraction)| {
            let strategies = RankerKind::ALL
                .iter()
                .enumerate()
                .map(|(s_idx, kind)| metrics_grid[b_idx][s_idx].finalize(kind.display_name()))
                .collect();
            BudgetEvalReport {
                budget_fraction,
                strategies,
            }
        })
        .collect();

    Ok(WalkForwardReport {
        total_runs: sorted_runs.len(),
        evaluated_runs,
        failing_runs,
        min_train_runs: start_idx,
        retrain_every,
        gbdt_steps,
        zero_leakage_verified,
        budgets,
    })
}
