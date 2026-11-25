use pts_core::{Config, SelectionReason};
use serde::{Deserialize, Serialize};

/// Minimum effective test duration floor in milliseconds.
/// Clamping prevents sub-millisecond tests from dominating `prob / eff_ms` yield ratios.
pub const MIN_EFF_DURATION_MS: f64 = 50.0;

/// Fallback per-test duration estimate in milliseconds when a repository has no timing history.
pub const DEFAULT_MEDIAN_DURATION_MS: f64 = 200.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateTest {
    pub test_id: String,
    pub prob: f64,
    pub est_duration_ms: Option<f64>,
    pub days_since_last_run: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    Budget {
        budget_ms: u64,
    },
    TopK {
        k: usize,
    },
    TargetRecall {
        target_recall: f64,
        max_budget_ms: Option<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedTest {
    pub test_id: String,
    pub predicted_prob: f64,
    pub est_duration_ms: u64,
    pub yield_score: f64,
    pub selection_reason: SelectionReason,
    pub sampling_propensity: f64,
}

#[derive(Debug, Clone)]
struct ScoredCandidate {
    test_id: String,
    prob: f64,
    eff_ms: f64,
    yield_score: f64,
    days_since_last_run: f64,
}

/// Deterministic SplitMix64 PRNG seeded from the first 8 bytes of `diff_fingerprint`.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn from_fingerprint(diff_fingerprint: &str) -> Self {
        let seed = seed_from_fingerprint(diff_fingerprint);
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
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

/// Pure, deterministic cost-aware test selector.
/// All score ties are broken lexicographically by `test_id` ascending.
pub fn select_tests(
    candidates: &[CandidateTest],
    strategy: SelectionStrategy,
    diff_fingerprint: &str,
    median_repo_test_ms: Option<f64>,
    config: &Config,
    primary_reason: SelectionReason,
) -> Vec<SelectedTest> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let fallback_ms = median_repo_test_ms
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_MEDIAN_DURATION_MS);

    let scored: Vec<ScoredCandidate> = candidates
        .iter()
        .map(|c| {
            let prob = if c.prob.is_finite() {
                c.prob.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let eff_ms = c
                .est_duration_ms
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(fallback_ms)
                .max(MIN_EFF_DURATION_MS);
            let days = if c.days_since_last_run.is_finite() && c.days_since_last_run > 0.0 {
                c.days_since_last_run
            } else {
                0.0
            };
            ScoredCandidate {
                test_id: c.test_id.clone(),
                prob,
                eff_ms,
                yield_score: prob / eff_ms,
                days_since_last_run: days,
            }
        })
        .collect();

    match strategy {
        SelectionStrategy::Budget { budget_ms } => select_budget(
            scored,
            budget_ms,
            diff_fingerprint,
            config.explore_fraction,
            primary_reason,
        ),
        SelectionStrategy::TopK { k } => select_top_k(scored, k, primary_reason),
        SelectionStrategy::TargetRecall {
            target_recall,
            max_budget_ms,
        } => select_target_recall(scored, target_recall, max_budget_ms, primary_reason),
    }
}

fn select_budget(
    mut candidates: Vec<ScoredCandidate>,
    budget_ms: u64,
    diff_fingerprint: &str,
    explore_fraction: f64,
    primary_reason: SelectionReason,
) -> Vec<SelectedTest> {
    let frac = if explore_fraction.is_finite() {
        explore_fraction.clamp(0.0, 1.0)
    } else {
        0.10
    };
    let explore_budget_ms = (budget_ms as f64 * frac) as u64;
    let exploit_budget_ms = budget_ms.saturating_sub(explore_budget_ms);

    candidates.sort_by(|a, b| {
        b.yield_score
            .total_cmp(&a.yield_score)
            .then_with(|| a.test_id.cmp(&b.test_id))
    });

    let mut selected = Vec::new();
    let mut unselected = Vec::new();
    let mut exploit_spent_ms = 0.0_f64;

    for cand in candidates {
        if selected.is_empty() || exploit_spent_ms + cand.eff_ms <= exploit_budget_ms as f64 {
            exploit_spent_ms += cand.eff_ms;
            selected.push(SelectedTest {
                test_id: cand.test_id,
                predicted_prob: cand.prob,
                est_duration_ms: cand.eff_ms.round() as u64,
                yield_score: cand.yield_score,
                selection_reason: primary_reason,
                sampling_propensity: 1.0,
            });
        } else {
            unselected.push(cand);
        }
    }

    if explore_budget_ms == 0 || unselected.is_empty() {
        return selected;
    }

    // Canonicalize unselected order by test_id so exploration is independent of input order.
    unselected.sort_by(|a, b| a.test_id.cmp(&b.test_id));

    let mut rng = SplitMix64::from_fingerprint(diff_fingerprint);
    let mut explore_spent_ms = 0.0_f64;

    loop {
        let remaining_budget = (explore_budget_ms as f64) - explore_spent_ms;
        if remaining_budget <= 0.0 {
            break;
        }

        let eligible_indices: Vec<usize> = unselected
            .iter()
            .enumerate()
            .filter(|(_, c)| c.eff_ms <= remaining_budget)
            .map(|(idx, _)| idx)
            .collect();

        if eligible_indices.is_empty() {
            break;
        }

        let weights: Vec<f64> = eligible_indices
            .iter()
            .map(|&idx| 1.0 + unselected[idx].days_since_last_run)
            .collect();
        let sum_w: f64 = weights.iter().sum();
        if sum_w <= 0.0 {
            break;
        }

        let draw = rng.next_f64() * sum_w;
        let mut cumulative = 0.0;
        let mut chosen_pos = eligible_indices.len() - 1;
        for (pos, &w) in weights.iter().enumerate() {
            cumulative += w;
            if draw < cumulative {
                chosen_pos = pos;
                break;
            }
        }

        let chosen_idx = eligible_indices[chosen_pos];
        let chosen_weight = weights[chosen_pos];
        let propensity = (chosen_weight / sum_w).clamp(0.01, 1.0);
        let cand = unselected.remove(chosen_idx);

        explore_spent_ms += cand.eff_ms;
        selected.push(SelectedTest {
            test_id: cand.test_id,
            predicted_prob: cand.prob,
            est_duration_ms: cand.eff_ms.round() as u64,
            yield_score: cand.yield_score,
            selection_reason: SelectionReason::Explore,
            sampling_propensity: propensity,
        });
    }

    selected
}

fn select_top_k(
    mut candidates: Vec<ScoredCandidate>,
    k: usize,
    primary_reason: SelectionReason,
) -> Vec<SelectedTest> {
    if k == 0 {
        return Vec::new();
    }

    candidates.sort_by(|a, b| {
        b.prob
            .total_cmp(&a.prob)
            .then_with(|| a.test_id.cmp(&b.test_id))
    });

    candidates
        .into_iter()
        .take(k)
        .map(|cand| SelectedTest {
            test_id: cand.test_id,
            predicted_prob: cand.prob,
            est_duration_ms: cand.eff_ms.round() as u64,
            yield_score: cand.yield_score,
            selection_reason: primary_reason,
            sampling_propensity: 1.0,
        })
        .collect()
}

fn select_target_recall(
    mut candidates: Vec<ScoredCandidate>,
    target_recall: f64,
    max_budget_ms: Option<u64>,
    primary_reason: SelectionReason,
) -> Vec<SelectedTest> {
    let target = target_recall.clamp(0.0, 1.0);

    candidates.sort_by(|a, b| {
        b.prob
            .total_cmp(&a.prob)
            .then_with(|| a.test_id.cmp(&b.test_id))
    });

    let n = candidates.len();
    let mut suffix_pass = vec![1.0_f64; n + 1];
    for i in (0..n).rev() {
        suffix_pass[i] = suffix_pass[i + 1] * (1.0 - candidates[i].prob);
    }

    let mut selected = Vec::new();
    let mut spent_ms = 0.0_f64;

    for (i, cand) in candidates.into_iter().enumerate() {
        let miss_prob = 1.0 - suffix_pass[i];
        if miss_prob <= 1.0 - target {
            break;
        }
        if let Some(cap_ms) = max_budget_ms
            && !selected.is_empty()
            && spent_ms + cand.eff_ms > cap_ms as f64
        {
            break;
        }
        spent_ms += cand.eff_ms;
        selected.push(SelectedTest {
            test_id: cand.test_id,
            predicted_prob: cand.prob,
            est_duration_ms: cand.eff_ms.round() as u64,
            yield_score: cand.yield_score,
            selection_reason: primary_reason,
            sampling_propensity: 1.0,
        });
    }

    selected
}
