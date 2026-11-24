use serde::{Deserialize, Serialize};

const PLATT_LAMBDA: f64 = 1e-4;
const PLATT_MAX_ITER: usize = 50;
const PLATT_TOL: f64 = 1e-7;
const BASE_RATE_EPS: f64 = 1e-4;
const LOGIT_EPS: f64 = 1e-12;

/// Platt scaling probability calibrator:
/// `P(y = 1 | z) = sigmoid(a * z + b)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlattCalibrator {
    pub a: f64,
    pub b: f64,
}

impl PlattCalibrator {
    /// Deterministic fallback calibrator used on degenerate splits or non-monotonic fits:
    /// `a = 1.0`, `b = logit(train_base_rate.clamp(1e-4, 1.0 - 1e-4))`.
    pub fn fallback(train_base_rate: f64) -> Self {
        let rate = if train_base_rate.is_finite() {
            train_base_rate.clamp(BASE_RATE_EPS, 1.0 - BASE_RATE_EPS)
        } else {
            BASE_RATE_EPS
        };
        Self {
            a: 1.0,
            b: logit(rate),
        }
    }

    /// Fits Platt scaling parameters `(a, b)` on unweighted calibration `(raw_logit, is_positive)`
    /// pairs using Platt (1999) target smoothing, L2 regularization (`lambda = 1e-4`), and
    /// damped 2x2 Newton-Raphson with Armijo backtracking line search.
    ///
    /// Guardrails:
    /// - If calibration has 0 positives or 0 negatives, falls back to `Self::fallback(train_base_rate)`.
    /// - If optimization yields `a < 0.0` or non-finite values, falls back to `Self::fallback(train_base_rate)`.
    pub fn fit(logits: &[f64], labels: &[bool], train_base_rate: f64) -> Self {
        if logits.len() != labels.len() || logits.is_empty() {
            return Self::fallback(train_base_rate);
        }

        let mut pairs: Vec<(f64, bool)> = Vec::with_capacity(logits.len());
        let mut n_pos: usize = 0;
        let mut n_neg: usize = 0;

        for (&z, &y) in logits.iter().zip(labels.iter()) {
            if !z.is_finite() {
                continue;
            }
            if y {
                n_pos += 1;
            } else {
                n_neg += 1;
            }
            pairs.push((z, y));
        }

        if n_pos == 0 || n_neg == 0 {
            return Self::fallback(train_base_rate);
        }

        // Platt (1999) smoothed targets.
        let t_pos = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let t_neg = 1.0 / (n_neg as f64 + 2.0);

        let targets: Vec<f64> = pairs
            .iter()
            .map(|&(_, y)| if y { t_pos } else { t_neg })
            .collect();
        let zs: Vec<f64> = pairs.iter().map(|&(z, _)| z).collect();

        // Initialize at slope 1.0 and smoothed calibration prior intercept.
        let mut a = 1.0_f64;
        let mut b = ((n_pos as f64 + 1.0) / (n_neg as f64 + 1.0)).ln();
        let mut loss = objective(&zs, &targets, a, b);

        for _ in 0..PLATT_MAX_ITER {
            let (ga, gb, haa, hab, hbb) = grad_hess(&zs, &targets, a, b);
            if !ga.is_finite() || !gb.is_finite() {
                return Self::fallback(train_base_rate);
            }
            if ga.abs().max(gb.abs()) < PLATT_TOL {
                break;
            }

            // Damped 2x2 Newton step (Levenberg-Marquardt damping if near-singular).
            let mut damping = 0.0_f64;
            let (da, db) = loop {
                let d_aa = haa + damping;
                let d_bb = hbb + damping;
                let det = d_aa * d_bb - hab * hab;
                if det > 1e-12 && det.is_finite() {
                    let step_a = -(d_bb * ga - hab * gb) / det;
                    let step_b = -(-hab * ga + d_aa * gb) / det;
                    if step_a.is_finite() && step_b.is_finite() {
                        break (step_a, step_b);
                    }
                }
                damping = if damping == 0.0 { 1e-6 } else { damping * 10.0 };
                if damping > 1e12 {
                    return Self::fallback(train_base_rate);
                }
            };

            let directional_deriv = ga * da + gb * db;
            if !directional_deriv.is_finite() || directional_deriv >= 0.0 {
                break;
            }

            // Backtracking line search (Armijo sufficient decrease).
            let mut step_size = 1.0_f64;
            let mut accepted = false;
            while step_size >= 1e-10 {
                let cand_a = a + step_size * da;
                let cand_b = b + step_size * db;
                let cand_loss = objective(&zs, &targets, cand_a, cand_b);
                if cand_loss.is_finite() && cand_loss <= loss + 1e-4 * step_size * directional_deriv
                {
                    let delta = (cand_a - a).abs().max((cand_b - b).abs());
                    a = cand_a;
                    b = cand_b;
                    loss = cand_loss;
                    accepted = true;
                    if delta < PLATT_TOL {
                        break;
                    }
                    break;
                }
                step_size *= 0.5;
            }

            if !accepted || (step_size * da).abs().max((step_size * db).abs()) < PLATT_TOL {
                break;
            }
        }

        // Monotonicity & numerical sanity guardrail.
        if !a.is_finite() || !b.is_finite() || a < 0.0 {
            return Self::fallback(train_base_rate);
        }

        Self { a, b }
    }

    /// Predicts calibrated failure probability in `(0.0, 1.0)` from a raw log-odds margin.
    pub fn predict_prob(&self, raw_logit: f64) -> f64 {
        let z = if raw_logit.is_finite() {
            raw_logit
        } else {
            0.0
        };
        sigmoid(self.a * z + self.b)
    }
}

/// Numerically stable logistic sigmoid `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Numerically stable log-odds `ln(p / (1 - p))` with clamping to avoid infinities.
pub fn logit(p: f64) -> f64 {
    let clamped = if p.is_finite() {
        p.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    } else {
        0.5
    };
    (clamped / (1.0 - clamped)).ln()
}

/// Computes mean squared probability error (Brier score).
pub fn brier_score(probs: &[f64], labels: &[bool]) -> f64 {
    if probs.is_empty() || probs.len() != labels.len() {
        return 0.0;
    }
    let sum_sq: f64 = probs
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| {
            let target = if y { 1.0 } else { 0.0 };
            let diff = p.clamp(0.0, 1.0) - target;
            diff * diff
        })
        .sum();
    sum_sq / (probs.len() as f64)
}

fn softplus(x: f64) -> f64 {
    if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn objective(zs: &[f64], targets: &[f64], a: f64, b: f64) -> f64 {
    let mut nll = 0.0_f64;
    for (&z, &t) in zs.iter().zip(targets.iter()) {
        let s = a * z + b;
        nll += softplus(s) - t * s;
    }
    nll + 0.5 * PLATT_LAMBDA * (a * a + b * b)
}

fn grad_hess(zs: &[f64], targets: &[f64], a: f64, b: f64) -> (f64, f64, f64, f64, f64) {
    let mut ga = PLATT_LAMBDA * a;
    let mut gb = PLATT_LAMBDA * b;
    let mut haa = PLATT_LAMBDA;
    let mut hab = 0.0_f64;
    let mut hbb = PLATT_LAMBDA;

    for (&z, &t) in zs.iter().zip(targets.iter()) {
        let s = a * z + b;
        let p = sigmoid(s);
        let diff = p - t;
        let w = (p * (1.0 - p)).max(1e-12);

        ga += diff * z;
        gb += diff;
        haa += w * z * z;
        hab += w * z;
        hbb += w;
    }

    (ga, gb, haa, hab, hbb)
}
