use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};
use pts_core::{CommitRecord, HistoricalRun, PtsError, TestStatus};
use serde::{Deserialize, Serialize};

use crate::accumulator::{HistoryAccumulator, TrainingRow};
use crate::features::{FEATURE_NAMES, NUM_FEATURES};
use crate::platt::{PlattCalibrator, brier_score, logit, sigmoid};

pub const MODEL_ARTIFACT_VERSION: u32 = 1;

/// Verified against `forust-ml 0.8.1`: `GradientBooster::predict` outputs additive
/// raw log-odds margins (`base_score + sum(tree_outputs)`), not sigmoid probabilities.
pub const BOOSTER_PREDICTS_RAW_MARGINS: bool = true;

/// SPEC.md Section 5.3 v1 GBDT hyperparameters.
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

/// Strict chronological 80/20 split at a run timestamp boundary (`max_train_ts < min_cal_ts`).
#[derive(Debug, Clone)]
pub struct ChronologicalSplit {
    pub train_rows: Vec<TrainingRow>,
    pub cal_rows: Vec<TrainingRow>,
    pub max_train_ts: i64,
    pub min_cal_ts: i64,
    pub train_run_count: usize,
    pub cal_run_count: usize,
}

/// Builds leakage-free chronological features via `HistoryAccumulator` and partitions runs
/// 80/20 at the run boundary, guaranteeing strict `max_train_ts < min_cal_ts` and zero
/// `run_id` overlap between train and calibration sets.
pub fn build_chronological_split(
    runs: &[HistoricalRun],
    commits: &[CommitRecord],
) -> Result<ChronologicalSplit, PtsError> {
    let dataset = HistoryAccumulator::build_dataset(runs, commits);
    if dataset.is_empty() {
        return Err(PtsError::InvalidInput(
            "no non-skip test outcomes available for training".to_string(),
        ));
    }

    // Collect distinct runs present in the non-skip dataset in chronological order.
    let mut run_order: Vec<(i64, String)> = Vec::new();
    for row in &dataset {
        let key = (row.started_at, row.run_id.clone());
        if run_order.last() != Some(&key) {
            run_order.push(key);
        }
    }

    if run_order.len() < 2 {
        return Err(PtsError::InvalidInput(
            "at least 2 historical runs with distinct timestamps are required for 80/20 train/cal split"
                .to_string(),
        ));
    }

    let total_runs = run_order.len();
    let target_train = ((total_runs as f64) * 0.80).floor() as usize;
    let target_train = target_train.clamp(1, total_runs - 1);

    // Ensure the split falls strictly between distinct timestamps (`max_train_ts < min_cal_ts`).
    let split_idx = if run_order[target_train - 1].0 < run_order[target_train].0 {
        target_train
    } else {
        // Try moving right past the timestamp tie group.
        let mut right = target_train;
        while right < total_runs && run_order[right].0 == run_order[target_train - 1].0 {
            right += 1;
        }
        if right < total_runs {
            right
        } else {
            // Otherwise move left before the timestamp tie group.
            let mut left = target_train;
            while left > 0 && run_order[left - 1].0 == run_order[target_train].0 {
                left -= 1;
            }
            if left > 0 {
                left
            } else {
                return Err(PtsError::InvalidInput(
                    "cannot create leakage-free chronological split: all runs share identical started_at timestamp"
                        .to_string(),
                ));
            }
        }
    };

    let max_train_ts = run_order[split_idx - 1].0;
    let min_cal_ts = run_order[split_idx].0;
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
        return Err(PtsError::InvalidInput(
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
    })
}

/// Serializable model artifact persisted to `.pts/model.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub version: u32,
    pub trained_at: i64,
    pub max_train_ts: i64,
    pub feature_names: Vec<String>,
    pub booster_json: String,
    pub platt_a: f64,
    pub platt_b: f64,
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
    pub fn from_artifact(artifact: ModelArtifact) -> Result<Self, PtsError> {
        if artifact.version != MODEL_ARTIFACT_VERSION {
            return Err(PtsError::Parse(format!(
                "unsupported model artifact version {} (expected {})",
                artifact.version, MODEL_ARTIFACT_VERSION
            )));
        }
        let expected_names: Vec<String> = FEATURE_NAMES.iter().map(|s| (*s).to_string()).collect();
        if artifact.feature_names != expected_names {
            return Err(PtsError::Parse(
                "model artifact feature_names do not match current 16-feature contract".to_string(),
            ));
        }
        if !artifact.platt_a.is_finite()
            || !artifact.platt_b.is_finite()
            || artifact.platt_a < 0.0
        {
            return Err(PtsError::Parse(
                "model artifact contains invalid Platt calibration parameters".to_string(),
            ));
        }

        let booster = GradientBooster::from_json(&artifact.booster_json)
            .map_err(|e| PtsError::Parse(format!("failed to deserialize forust-ml booster: {e}")))?;
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

    pub fn from_json(json: &str) -> Result<Self, PtsError> {
        let artifact: ModelArtifact = serde_json::from_str(json)
            .map_err(|e| PtsError::Parse(format!("failed to parse model artifact JSON: {e}")))?;
        Self::from_artifact(artifact)
    }

    pub fn to_json(&self) -> Result<String, PtsError> {
        serde_json::to_string_pretty(&self.artifact)
            .map_err(|e| PtsError::Parse(format!("failed to serialize model artifact JSON: {e}")))
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
    pub min_cal_ts: i64,
    pub uncalibrated_brier: f64,
    pub calibrated_brier: f64,
}

/// Builds a configured `forust-ml` `GradientBooster` with SPEC.md Section 5.3 defaults.
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
) -> Result<TrainResult, PtsError> {
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
        return Err(PtsError::InvalidInput(format!(
            "training split requires both passing and failing outcomes (got {train_pos} positive, {train_neg} negative across {} train rows)",
            train_labels.len()
        )));
    }

    let train_col_major = to_column_major(&train_features);
    let train_matrix = Matrix::new(&train_col_major, train_features.len(), NUM_FEATURES);

    let mut booster = build_default_booster();
    booster
        .fit(&train_matrix, &train_labels, &train_weights, None)
        .map_err(|e| PtsError::InvalidInput(format!("forust-ml training failed: {e}")))?;

    // Calibrate on unweighted chronological holdout runs.
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

    let train_base_rate = train_pos as f64 / (train_labels.len() as f64);
    let calibrator = PlattCalibrator::fit(&cal_raw_margins, &cal_labels_bool, train_base_rate);

    let uncalibrated_probs: Vec<f64> = cal_raw_margins.iter().map(|&z| sigmoid(z)).collect();
    let calibrated_probs: Vec<f64> = cal_raw_margins
        .iter()
        .map(|&z| calibrator.predict_prob(z))
        .collect();

    let uncalibrated_brier = brier_score(&uncalibrated_probs, &cal_labels_bool);
    let calibrated_brier = brier_score(&calibrated_probs, &cal_labels_bool);

    let booster_json = booster
        .json_dump()
        .map_err(|e| PtsError::Parse(format!("failed to serialize forust-ml booster: {e}")))?;

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
