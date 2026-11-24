use serde::{Deserialize, Serialize};

/// Default number of equal-width reliability diagram bins over `[0.0, 1.0]`.
pub const NUM_CALIBRATION_BINS: usize = 10;

/// Single equal-width probability bin in `[0.0, 1.0]` for calibration reliability diagrams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub bin_center: f64,
    pub mean_predicted: f64,
    pub empirical_positive_rate: f64,
    pub count: usize,
}

/// Reliability diagram bins comparing raw GBDT sigmoid probabilities against Platt-calibrated probabilities.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBins {
    pub raw_gbdt: Vec<CalibrationBin>,
    pub platt_calibrated: Vec<CalibrationBin>,
}

/// Aggregated walk-forward evaluation metrics for a single strategy at a given time budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy: String,
    pub evaluated_runs: usize,
    pub failing_runs: usize,
    pub detected_failing_runs: usize,
    /// Commit-Level Failure Detection Rate: fraction of failing runs where at least
    /// one failing test was selected.
    pub commit_fdr: f64,
    pub total_failures: usize,
    pub caught_failures: usize,
    /// Individual Failure Recall: fraction of total failed test executions caught.
    pub individual_recall: f64,
    pub selected_duration_ms: u64,
    pub full_suite_duration_ms: u64,
    /// Wall-Clock Time Saved: `1.0 - (selected_duration_ms / full_suite_duration_ms)`.
    pub wall_clock_time_saved: f64,
    /// Mean squared probability error across all evaluated non-skip outcomes.
    pub brier_score: f64,
    /// Raw GBDT sigmoid Brier score prior to Platt calibration (populated for `PTS (GBDT+Platt)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncalibrated_brier_score: Option<f64>,
    /// Average precision (PR-AUC) across all evaluated non-skip outcomes.
    pub pr_auc: f64,
}

/// Running accumulator for a single `(budget, strategy)` cell during walk-forward evaluation.
#[derive(Debug, Clone, Default)]
pub struct MetricsAccumulator {
    evaluated_runs: usize,
    failing_runs: usize,
    detected_failing_runs: usize,
    total_failures: usize,
    caught_failures: usize,
    selected_duration_ms: u64,
    full_suite_duration_ms: u64,
    predictions: Vec<(f64, bool)>,
}

impl MetricsAccumulator {
    pub fn record_run(
        &mut self,
        run_total_failures: usize,
        run_caught_failures: usize,
        run_selected_duration_ms: u64,
        run_full_duration_ms: u64,
        run_predictions: &[(f64, bool)],
    ) {
        self.evaluated_runs += 1;
        if run_total_failures > 0 {
            self.failing_runs += 1;
            if run_caught_failures > 0 {
                self.detected_failing_runs += 1;
            }
        }
        self.total_failures += run_total_failures;
        self.caught_failures += run_caught_failures;
        self.selected_duration_ms = self
            .selected_duration_ms
            .saturating_add(run_selected_duration_ms);
        self.full_suite_duration_ms = self
            .full_suite_duration_ms
            .saturating_add(run_full_duration_ms);
        self.predictions.extend_from_slice(run_predictions);
    }

    pub fn finalize(&self, strategy: &str) -> StrategyMetrics {
        let commit_fdr = if self.failing_runs == 0 {
            0.0
        } else {
            self.detected_failing_runs as f64 / self.failing_runs as f64
        };
        let individual_recall = if self.total_failures == 0 {
            0.0
        } else {
            self.caught_failures as f64 / self.total_failures as f64
        };
        let wall_clock_time_saved = if self.full_suite_duration_ms == 0 {
            0.0
        } else {
            1.0 - (self.selected_duration_ms as f64 / self.full_suite_duration_ms as f64)
        };
        let brier_score = compute_brier_score(&self.predictions);
        let pr_auc = compute_pr_auc(&self.predictions);

        StrategyMetrics {
            strategy: strategy.to_string(),
            evaluated_runs: self.evaluated_runs,
            failing_runs: self.failing_runs,
            detected_failing_runs: self.detected_failing_runs,
            commit_fdr,
            total_failures: self.total_failures,
            caught_failures: self.caught_failures,
            individual_recall,
            selected_duration_ms: self.selected_duration_ms,
            full_suite_duration_ms: self.full_suite_duration_ms,
            wall_clock_time_saved,
            brier_score,
            uncalibrated_brier_score: None,
            pr_auc,
        }
    }
}

/// Partitions `(predicted_prob, is_failure)` pairs into `num_bins` equal-width bins over `[0.0, 1.0]`.
pub fn compute_calibration_bins(
    predictions: &[(f64, bool)],
    num_bins: usize,
) -> Vec<CalibrationBin> {
    if num_bins == 0 {
        return Vec::new();
    }

    let mut counts = vec![0usize; num_bins];
    let mut sum_preds = vec![0.0_f64; num_bins];
    let mut pos_counts = vec![0usize; num_bins];

    for &(p, y) in predictions {
        let clamped = if p.is_finite() {
            p.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let bin_idx = ((clamped * num_bins as f64).floor() as usize).min(num_bins - 1);
        counts[bin_idx] += 1;
        sum_preds[bin_idx] += clamped;
        if y {
            pos_counts[bin_idx] += 1;
        }
    }

    (0..num_bins)
        .map(|idx| {
            let count = counts[idx];
            let bin_center = (idx as f64 + 0.5) / (num_bins as f64);
            let (mean_predicted, empirical_positive_rate) = if count > 0 {
                (
                    sum_preds[idx] / (count as f64),
                    pos_counts[idx] as f64 / (count as f64),
                )
            } else {
                (0.0, 0.0)
            };
            CalibrationBin {
                bin_center,
                mean_predicted,
                empirical_positive_rate,
                count,
            }
        })
        .collect()
}

/// Computes mean squared probability error (Brier score) over `(predicted_prob, is_failure)` pairs.
pub fn compute_brier_score(predictions: &[(f64, bool)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = predictions
        .iter()
        .map(|&(p, y)| {
            let clamped = if p.is_finite() {
                p.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let target = if y { 1.0 } else { 0.0 };
            let diff = clamped - target;
            diff * diff
        })
        .sum();
    sum_sq / (predictions.len() as f64)
}

/// Computes Average Precision (PR-AUC) matching step-function precision-recall integration,
/// grouping tied probability scores so constant predictions receive exactly base-rate AP.
pub fn compute_pr_auc(predictions: &[(f64, bool)]) -> f64 {
    let total_pos = predictions.iter().filter(|&&(_, y)| y).count();
    if total_pos == 0 || predictions.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<(f64, bool)> = predictions
        .iter()
        .map(|&(p, y)| (if p.is_finite() { p } else { 0.0 }, y))
        .collect();
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut cum_tp = 0usize;
    let mut cum_fp = 0usize;
    let mut prev_recall = 0.0_f64;
    let mut ap = 0.0_f64;

    let mut i = 0;
    while i < sorted.len() {
        let score = sorted[i].0;
        let mut j = i;
        while j < sorted.len() && sorted[j].0.total_cmp(&score).is_eq() {
            if sorted[j].1 {
                cum_tp += 1;
            } else {
                cum_fp += 1;
            }
            j += 1;
        }

        let recall = cum_tp as f64 / total_pos as f64;
        let precision = cum_tp as f64 / (cum_tp + cum_fp) as f64;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
        i = j;
    }

    ap
}
