use failcast_core::{Config, SelectionReason};
use failcast_ml::features::{
    IDX_AST_MIN_HOP, IDX_DIRECT_TEST_MODIFIED, IDX_EWMA_FAIL_10, NUM_FEATURES,
};
use failcast_ml::{CandidateTest, SelectedTest, SelectionStrategy, ast_hop_decay, select_tests};
use serde::{Deserialize, Serialize};

/// Identifiers for the evaluated selection strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RankerKind {
    FailcastGbdt,
    Heuristic,
    StaticAst,
    RecentFailureRate,
    Random,
}

impl RankerKind {
    pub const ALL: [RankerKind; 5] = [
        RankerKind::FailcastGbdt,
        RankerKind::Heuristic,
        RankerKind::StaticAst,
        RankerKind::RecentFailureRate,
        RankerKind::Random,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::FailcastGbdt => "Failcast (GBDT+Platt)",
            Self::Heuristic => "Heuristic",
            Self::StaticAst => "StaticAst",
            Self::RecentFailureRate => "RecentFailureRate",
            Self::Random => "Random",
        }
    }
}

/// Deterministic SplitMix64 PRNG seeded from the first 8 bytes of `diff_fingerprint`.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn from_fingerprint(diff_fingerprint: &str) -> Self {
        Self {
            state: seed_from_fingerprint(diff_fingerprint),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        // Keep strictly in (0, 1) so random scores are always valid probabilities.
        u.clamp(1e-6, 1.0 - 1e-6)
    }
}

fn seed_from_fingerprint(diff_fingerprint: &str) -> u64 {
    if diff_fingerprint.len() >= 16
        && let Ok(parsed) = u64::from_str_radix(&diff_fingerprint[..16], 16)
    {
        return parsed;
    }

    let mut bytes = [0u8; 8];
    for (slot, b) in bytes.iter_mut().zip(diff_fingerprint.as_bytes().iter()) {
        *slot = *b;
    }
    u64::from_be_bytes(bytes)
}

/// Baseline 1 (`Random`): deterministic SplitMix64 uniform scores seeded per `diff_fingerprint`,
/// invariant to input test ordering (canonicalized by `test_id` ascending).
pub fn score_random_batch(test_ids: &[String], diff_fingerprint: &str) -> Vec<f64> {
    let mut order: Vec<usize> = (0..test_ids.len()).collect();
    order.sort_by(|&a, &b| test_ids[a].cmp(&test_ids[b]));

    let mut rng = SplitMix64::from_fingerprint(diff_fingerprint);
    let mut scores = vec![0.0_f64; test_ids.len()];
    for idx in order {
        scores[idx] = rng.next_f64();
    }
    scores
}

/// Baseline 2 (`StaticAst`): ranks strictly by AST import-graph hop distance +
/// direct test file modification, ignoring historical failure/churn signals.
pub fn score_static_ast(features: &[f64; NUM_FEATURES]) -> f64 {
    let direct = if features[IDX_DIRECT_TEST_MODIFIED].is_nan() {
        0.0
    } else {
        features[IDX_DIRECT_TEST_MODIFIED]
    };
    let ast_decay = ast_hop_decay(features[IDX_AST_MIN_HOP]);
    let raw = 3.0 * direct + 2.5 * ast_decay;
    1.0 / (1.0 + (-(raw - 1.5)).exp())
}

/// Baseline 3 (`RecentFailureRate`): ranks strictly by `ewma_fail_10`, ignoring diff features.
pub fn score_recent_failure_rate(features: &[f64; NUM_FEATURES]) -> f64 {
    let ewma = features[IDX_EWMA_FAIL_10];
    if ewma.is_nan() {
        0.0
    } else {
        ewma.clamp(0.0, 1.0)
    }
}

/// Runs any strategy's predicted probabilities through the exact same cost-aware
/// budget selector interface (`failcast_ml::select_tests`).
pub fn select_for_budget(
    candidates: &[CandidateTest],
    budget_ms: u64,
    diff_fingerprint: &str,
    median_repo_test_ms: Option<f64>,
    config: &Config,
    primary_reason: SelectionReason,
) -> Vec<SelectedTest> {
    select_tests(
        candidates,
        SelectionStrategy::Budget { budget_ms },
        diff_fingerprint,
        median_repo_test_ms,
        config,
        primary_reason,
    )
}
