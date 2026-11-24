use pts_core::{Config, HistorySnapshot};

use crate::features::{
    IDX_AST_MIN_HOP, IDX_COCHANGE_FREQ_MAX, IDX_DIRECT_TEST_MODIFIED, IDX_EWMA_FAIL_10,
    IDX_PATH_JACCARD_MAX, NUM_FEATURES,
};

/// Decays AST hop distance into `(0.0, 1.0]`; returns `0.0` for `NAN` or negative hops.
pub fn ast_hop_decay(ast_min_hop: f64) -> f64 {
    if ast_min_hop.is_nan() || ast_min_hop < 0.0 {
        0.0
    } else {
        1.0 / (1.0 + ast_min_hop)
    }
}

/// Cold-start deterministic heuristic ranker (SPEC.md Section 5.2).
/// Returns a bounded failure probability estimate in `(0.0, 1.0)`.
pub fn score_heuristic(features: &[f64; NUM_FEATURES]) -> f64 {
    let direct_test_modified = nan_to_zero(features[IDX_DIRECT_TEST_MODIFIED]);
    let ast_decay = ast_hop_decay(features[IDX_AST_MIN_HOP]);
    let path_jaccard_max = nan_to_zero(features[IDX_PATH_JACCARD_MAX]);
    let cochange_freq_max = nan_to_zero(features[IDX_COCHANGE_FREQ_MAX]);
    let ewma_fail_10 = nan_to_zero(features[IDX_EWMA_FAIL_10]);

    let raw = 3.0 * direct_test_modified
        + 2.5 * ast_decay
        + 1.5 * path_jaccard_max
        + 1.5 * cochange_freq_max
        + 1.0 * ewma_fail_10;
    1.0 / (1.0 + (-(raw - 1.5)).exp())
}

/// Returns `true` when cold-start heuristic scoring should be used instead of GBDT.
pub fn should_use_heuristic(
    history: &HistorySnapshot,
    config: &Config,
    has_trained_model: bool,
) -> bool {
    !has_trained_model
        || history.total_runs < config.min_runs_for_ml
        || history.total_failures < config.min_failures_for_ml
}

fn nan_to_zero(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v }
}
