use failcast_core::{CommitRecord, FailcastError, HistoricalRun, TestStatus};
use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};
use serde::{Deserialize, Serialize};

use crate::accumulator::{HistoryAccumulator, TrainingRow};
use crate::features::{FEATURE_NAMES, NUM_FEATURES};
use crate::platt::{
    MIN_CAL_FAILING_RUNS, MIN_CAL_POSITIVES, PlattCalibrator, brier_score, logit, sigmoid,
};

pub const MODEL_ARTIFACT_VERSION: u32 = 1;
const MIN_TRAIN_RUN_FRACTION: f64 = 0.60;

/// Verified against `forust-ml 0.8.1`: `GradientBooster::predict` outputs additive
/// raw log-odds margins (`base_score + sum(tree_outputs)`), not sigmoid probabilities.
pub const BOOSTER_PREDICTS_RAW_MARGINS: bool = true;

/// Default GBDT training hyperparameters.
pub const GBDT_SEED: u64 = 42;
pub const GBDT_ITERATIONS: usize = 150;
pub const GBDT_LEARNING_RATE: f32 = 0.05;
pub const GBDT_MAX_DEPTH: usize = 5;
pub const GBDT_MIN_LEAF_WEIGHT: f32 = 5.0;
pub const GBDT_L2: f32 = 2.0;

pub const MIN_SAMPLE_WEIGHT: f64 = 1.0;
pub const MAX_SAMPLE_WEIGHT: f64 = 10.0;

/// Converts row-major `[row][col]` feature slices into `forust-ml`'s required
/// column-major flat layout: `out[col * num_rows + row] = rows[row][col]`.
pub fn to_column_major(rows: &[[f64; NUM_FEATURES]]) -> Vec<f64> {
    let num_rows = rows.len();
    let mut data = vec![0.0_f64; num_rows * NUM_FEATURES];
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            data[col_idx * num_rows + row_idx] = val;
        }
    }
    data
}

/// Normalizes a raw `GradientBooster::predict` output to log-odds margin before Platt scaling.
pub fn normalize_to_log_odds(raw_pred: f64, predicts_raw_margin: bool) -> f64 {
    if predicts_raw_margin {
        raw_pred
    } else {
        logit(raw_pred)
    }
}

/// Inverse-propensity training weight: `(1.0 / sampling_propensity).clamp(1.0, 10.0)`.
pub fn inverse_propensity_weight(sampling_propensity: f64) -> f64 {
    if !sampling_propensity.is_finite() || sampling_propensity <= 0.0 {
        MAX_SAMPLE_WEIGHT
    } else {
        (1.0 / sampling_propensity).clamp(MIN_SAMPLE_WEIGHT, MAX_SAMPLE_WEIGHT)
    }
}

/// Strict chronological split at a run timestamp boundary (`max_train_ts < min_cal_ts`).
/// Defaults to 80/20 and expands `cal_rows` backward down to 60% train when needed to
/// satisfy Platt calibration support (`>= 5` positives across `>= 2` distinct failing runs).
#[derive(Debug, Clone)]
pub struct ChronologicalSplit {
    pub train_rows: Vec<TrainingRow>,
    pub cal_rows: Vec<TrainingRow>,
    pub max_train_ts: i64,
    pub min_cal_ts: i64,
    pub train_run_count: usize,
    pub cal_run_count: usize,
    pub cal_positives: usize,
    pub cal_failing_runs: usize,
}

/// Builds leakage-free chronological features via `HistoryAccumulator` and partitions runs
/// at a strict run timestamp boundary (`max_train_ts < min_cal_ts`) with zero `run_id` overlap.
///
/// Defaults to an 80/20 train/cal split. If the tail 20% slice has fewer than `MIN_CAL_POSITIVES`
/// (5) positive outcomes or fewer than `MIN_CAL_FAILING_RUNS` (2) distinct failing runs, the
/// calibration window expands backward run-by-run while strictly preserving `max_train_ts < min_cal_ts`,
/// keeping at least 60% of runs in `train_rows`, and retaining both classes in `train_rows`.
pub fn build_chronological_split(
    runs: &[HistoricalRun],
    commits: &[CommitRecord],
) -> Result<ChronologicalSplit, FailcastError> {
    let dataset = HistoryAccumulator::build_dataset(runs, commits);
    if dataset.is_empty() {
        return Err(FailcastError::InvalidInput(
            "no non-skip test outcomes available for training".to_string(),
        ));
    }

    struct RunStats {
        started_at: i64,
        run_id: String,
        pos_count: usize,
        neg_count: usize,
    }

    let mut run_order: Vec<RunStats> = Vec::new();
    for row in &dataset {
        let is_pos = row.label >= 0.5;
        let same_run = run_order
            .last()
            .is_some_and(|r| r.started_at == row.started_at && r.run_id == row.run_id);
        if !same_run {
            run_order.push(RunStats {
                started_at: row.started_at,
                run_id: row.run_id.clone(),
                pos_count: usize::from(is_pos),
                neg_count: usize::from(!is_pos),
            });
        } else if let Some(last) = run_order.last_mut() {
            if is_pos {
                last.pos_count += 1;
            } else {
                last.neg_count += 1;
            }
        }
    }

    if run_order.len() < 2 {
        return Err(FailcastError::InvalidInput(
            "at least 2 historical runs with distinct timestamps are required for 80/20 train/cal split"
                .to_string(),
        ));
    }

    let total_runs = run_order.len();
    let target_train = ((total_runs as f64) * 0.80).floor() as usize;
    let target_train = target_train.clamp(1, total_runs - 1);

    // Ensure the default 80/20 split falls strictly between distinct timestamps (`max_train_ts < min_cal_ts`).
    let default_split_idx = if run_order[target_train - 1].started_at
        < run_order[target_train].started_at
    {
        target_train
    } else {
        let mut right = target_train;
        while right < total_runs
            && run_order[right].started_at == run_order[target_train - 1].started_at
        {
            right += 1;
        }
        if right < total_runs {
            right
        } else {
            let mut left = target_train;
            while left > 0 && run_order[left - 1].started_at == run_order[target_train].started_at {
                left -= 1;
            }
            if left > 0 {
                left
            } else {
                return Err(FailcastError::InvalidInput(
                    "cannot create leakage-free chronological split: all runs share identical started_at timestamp"
                        .to_string(),
                ));
            }
        }
    };

    let cal_support = |idx: usize| -> (usize, usize) {
        let mut pos = 0usize;
        let mut failing_runs = 0usize;
        for r in &run_order[idx..] {
            if r.pos_count > 0 {
                pos += r.pos_count;
                failing_runs += 1;
            }
        }
        (pos, failing_runs)
    };
    let train_has_both_classes = |idx: usize| -> bool {
        let mut pos = 0usize;
        let mut neg = 0usize;
        for r in &run_order[..idx] {
            pos += r.pos_count;
            neg += r.neg_count;
        }
        pos > 0 && neg > 0
    };

    let mut split_idx = default_split_idx;
    let (default_cal_pos, default_cal_fail_runs) = cal_support(default_split_idx);
    if default_cal_pos < MIN_CAL_POSITIVES || default_cal_fail_runs < MIN_CAL_FAILING_RUNS {
        let min_train_runs = ((total_runs as f64) * MIN_TRAIN_RUN_FRACTION).ceil() as usize;
        let min_train_runs = min_train_runs.clamp(1, default_split_idx);
        for cand_idx in (min_train_runs..default_split_idx).rev() {
            if run_order[cand_idx - 1].started_at >= run_order[cand_idx].started_at {
                continue;
            }
            if !train_has_both_classes(cand_idx) {
                continue;
            }
            let (cand_pos, cand_fail_runs) = cal_support(cand_idx);
            if cand_pos >= MIN_CAL_POSITIVES && cand_fail_runs >= MIN_CAL_FAILING_RUNS {
                split_idx = cand_idx;
                break;
            }
        }
    }

    let (cal_positives, cal_failing_runs) = cal_support(split_idx);
    let max_train_ts = run_order[split_idx - 1].started_at;
    let min_cal_ts = run_order[split_idx].started_at;
    debug_assert!(max_train_ts < min_cal_ts);

    let mut train_rows = Vec::new();
    let mut cal_rows = Vec::new();
    for row in dataset {
        if row.started_at <= max_train_ts {
            train_rows.push(row);
        } else {
            cal_rows.push(row);
        }
    }

    if train_rows.is_empty() || cal_rows.is_empty() {
        return Err(FailcastError::InvalidInput(
            "chronological 80/20 split produced an empty train or calibration partition"
                .to_string(),
        ));
    }

    Ok(ChronologicalSplit {
        train_rows,
        cal_rows,
        max_train_ts,
        min_cal_ts,
        train_run_count: split_idx,
        cal_run_count: total_runs - split_idx,
        cal_positives,
        cal_failing_runs,
    })
}

/// Serializable model artifact persisted to `.failcast/model.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub version: u32,
    pub trained_at: i64,
    pub max_train_ts: i64,
    pub feature_names: Vec<String>,
    pub booster_json: String,
    pub platt_a: f64,
    pub platt_b: f64,
    #[serde(default)]
    pub calibration_fallback: bool,
    pub train_rows: usize,
    pub pos_rows: usize,
    pub brier_score: f64,
}

/// Loaded, ready-to-score GBDT + Platt calibration model.
pub struct TrainedModel {
    pub artifact: ModelArtifact,
    booster: GradientBooster,
    calibrator: PlattCalibrator,
}

impl TrainedModel {
    pub fn from_artifact(artifact: ModelArtifact) -> Result<Self, FailcastError> {
        if artifact.version != MODEL_ARTIFACT_VERSION {
            return Err(FailcastError::Parse(format!(
                "unsupported model artifact version {} (expected {})",
                artifact.version, MODEL_ARTIFACT_VERSION
            )));
        }
        let expected_names: Vec<String> = FEATURE_NAMES.iter().map(|s| (*s).to_string()).collect();
        if artifact.feature_names != expected_names {
            return Err(FailcastError::Parse(
                "model artifact feature_names do not match current 16-feature contract".to_string(),
            ));
        }
        if !artifact.platt_a.is_finite() || !artifact.platt_b.is_finite() || artifact.platt_a < 0.0
        {
            return Err(FailcastError::Parse(
                "model artifact contains invalid Platt calibration parameters".to_string(),
            ));
        }

        let booster = GradientBooster::from_json(&artifact.booster_json).map_err(|e| {
            FailcastError::Parse(format!("failed to deserialize forust-ml booster: {e}"))
        })?;
        let calibrator = PlattCalibrator {
            a: artifact.platt_a,
            b: artifact.platt_b,
        };

        Ok(Self {
            artifact,
            booster,
            calibrator,
        })
    }

    pub fn from_json(json: &str) -> Result<Self, FailcastError> {
        let artifact: ModelArtifact = serde_json::from_str(json).map_err(|e| {
            FailcastError::Parse(format!("failed to parse model artifact JSON: {e}"))
        })?;
        Self::from_artifact(artifact)
    }

    pub fn to_json(&self) -> Result<String, FailcastError> {
        serde_json::to_string_pretty(&self.artifact).map_err(|e| {
            FailcastError::Parse(format!("failed to serialize model artifact JSON: {e}"))
        })
    }

    pub fn calibrator(&self) -> PlattCalibrator {
        self.calibrator
    }

    /// Predicts raw log-odds margins from the underlying GBDT booster.
    pub fn predict_raw_margins(&self, rows: &[[f64; NUM_FEATURES]]) -> Vec<f64> {
        if rows.is_empty() {
            return Vec::new();
        }
        let col_major = to_column_major(rows);
        let matrix = Matrix::new(&col_major, rows.len(), NUM_FEATURES);
        self.booster
            .predict(&matrix, false)
            .into_iter()
            .map(|raw| normalize_to_log_odds(raw, BOOSTER_PREDICTS_RAW_MARGINS))
            .collect()
    }

    /// Predicts Platt-calibrated failure probabilities in `(0.0, 1.0)` for a batch of feature rows.
    pub fn predict_probs(&self, rows: &[[f64; NUM_FEATURES]]) -> Vec<f64> {
        self.predict_raw_margins(rows)
            .into_iter()
            .map(|z| self.calibrator.predict_prob(z))
            .collect()
    }

    /// Predicts Platt-calibrated failure probability for a single feature vector.
    pub fn predict_one(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        self.predict_probs(std::slice::from_ref(features))
            .into_iter()
            .next()
            .unwrap_or(0.5)
    }
}

/// Full training summary returned by `train_gbdt_model`.
pub struct TrainResult {
    pub model: TrainedModel,
    pub train_run_count: usize,
    pub cal_run_count: usize,
    pub cal_rows: usize,
    pub cal_pos_rows: usize,
    pub cal_failing_runs: usize,
    pub calibration_fallback: bool,
    pub min_cal_ts: i64,
    pub uncalibrated_brier: f64,
    pub calibrated_brier: f64,
}

/// Builds a configured `forust-ml` `GradientBooster` with default hyperparameters.
pub fn build_default_booster() -> GradientBooster {
    GradientBooster::default()
        .set_objective_type(ObjectiveType::LogLoss)
        .set_iterations(GBDT_ITERATIONS)
        .set_learning_rate(GBDT_LEARNING_RATE)
        .set_max_depth(GBDT_MAX_DEPTH)
        .set_min_leaf_weight(GBDT_MIN_LEAF_WEIGHT)
        .set_l2(GBDT_L2)
        .set_seed(GBDT_SEED)
        .set_parallel(false)
        .set_missing(f64::NAN)
        .set_initialize_base_score(true)
}

/// Trains a deterministic `forust-ml` GBDT + unweighted Platt calibrator on a strict
/// chronological 80/20 run-boundary split.
///
/// Pure function: takes `trained_at` explicitly (no `SystemTime::now()`, no filesystem I/O).
pub fn train_gbdt_model(
    runs: &[HistoricalRun],
    commits: &[CommitRecord],
    trained_at: i64,
) -> Result<TrainResult, FailcastError> {
    let split = build_chronological_split(runs, commits)?;

    let train_features: Vec<[f64; NUM_FEATURES]> =
        split.train_rows.iter().map(|r| r.features).collect();
    let train_labels: Vec<f64> = split.train_rows.iter().map(|r| r.label).collect();
    let train_weights: Vec<f64> = split
        .train_rows
        .iter()
        .map(|r| inverse_propensity_weight(r.sampling_propensity))
        .collect();

    let train_pos = train_labels.iter().filter(|&&y| y >= 0.5).count();
    let train_neg = train_labels.len() - train_pos;
    if train_pos == 0 || train_neg == 0 {
        return Err(FailcastError::InvalidInput(format!(
            "training split requires both passing and failing outcomes (got {train_pos} positive, {train_neg} negative across {} train rows)",
            train_labels.len()
        )));
    }

    let train_col_major = to_column_major(&train_features);
    let train_matrix = Matrix::new(&train_col_major, train_features.len(), NUM_FEATURES);

    let mut booster = build_default_booster();
    booster
        .fit(&train_matrix, &train_labels, &train_weights, None)
        .map_err(|e| FailcastError::InvalidInput(format!("forust-ml training failed: {e}")))?;

    let cal_features: Vec<[f64; NUM_FEATURES]> =
        split.cal_rows.iter().map(|r| r.features).collect();
    let cal_labels_bool: Vec<bool> = split.cal_rows.iter().map(|r| r.label >= 0.5).collect();
    let cal_pos_rows = cal_labels_bool.iter().filter(|&&y| y).count();

    let cal_col_major = to_column_major(&cal_features);
    let cal_matrix = Matrix::new(&cal_col_major, cal_features.len(), NUM_FEATURES);
    let cal_raw_margins: Vec<f64> = booster
        .predict(&cal_matrix, false)
        .into_iter()
        .map(|raw| normalize_to_log_odds(raw, BOOSTER_PREDICTS_RAW_MARGINS))
        .collect();

    let (calibrator, calibration_fallback) = PlattCalibrator::fit_with_guardrail(
        &cal_raw_margins,
        &cal_labels_bool,
        split.cal_failing_runs,
    );

    let uncalibrated_probs: Vec<f64> = cal_raw_margins.iter().map(|&z| sigmoid(z)).collect();
    let calibrated_probs: Vec<f64> = cal_raw_margins
        .iter()
        .map(|&z| calibrator.predict_prob(z))
        .collect();

    let uncalibrated_brier = brier_score(&uncalibrated_probs, &cal_labels_bool);
    let calibrated_brier = brier_score(&calibrated_probs, &cal_labels_bool);

    let booster_json = booster
        .json_dump()
        .map_err(|e| FailcastError::Parse(format!("failed to serialize forust-ml booster: {e}")))?;

    // max_train_ts records the latest run timestamp used anywhere in training/calibration
    // so downstream anti-leakage assertions verify strict separation against future eval runs.
    let max_model_ts = split
        .cal_rows
        .iter()
        .map(|r| r.started_at)
        .max()
        .unwrap_or(split.max_train_ts);

    let artifact = ModelArtifact {
        version: MODEL_ARTIFACT_VERSION,
        trained_at,
        max_train_ts: max_model_ts,
        feature_names: FEATURE_NAMES.iter().map(|s| (*s).to_string()).collect(),
        booster_json,
        platt_a: calibrator.a,
        platt_b: calibrator.b,
        calibration_fallback,
        train_rows: split.train_rows.len(),
        pos_rows: train_pos,
        brier_score: calibrated_brier,
    };

    let model = TrainedModel {
        artifact,
        booster,
        calibrator,
    };

    Ok(TrainResult {
        model,
        train_run_count: split.train_run_count,
        cal_run_count: split.cal_run_count,
        cal_rows: split.cal_rows.len(),
        cal_pos_rows,
        cal_failing_runs: split.cal_failing_runs,
        calibration_fallback,
        min_cal_ts: split.min_cal_ts,
        uncalibrated_brier,
        calibrated_brier,
    })
}

/// Counts non-skip failure outcomes across a slice of historical runs.
pub fn count_failures(runs: &[HistoricalRun]) -> usize {
    runs.iter()
        .flat_map(|r| r.outcomes.iter())
        .filter(|o| matches!(o.status, TestStatus::Fail | TestStatus::Error))
        .count()
}
