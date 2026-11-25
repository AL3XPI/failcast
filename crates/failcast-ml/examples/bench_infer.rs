//! Pure `failcast-ml` GBDT + Platt inference throughput micro-benchmark.
//!
//! Run in release mode:
//!   `cargo run --release --example bench_infer -p failcast-ml`

use std::hint::black_box;
use std::time::Instant;

use failcast_core::{
    CommitRecord, DiffFile, DiffSummary, HistoricalOutcome, HistoricalRun, TestStatus,
};
use failcast_ml::{NUM_FEATURES, train_gbdt_model};

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

fn build_synthetic_training_history() -> (Vec<HistoricalRun>, Vec<CommitRecord>) {
    let mut rng = SplitMix64::new(0x5054_535F_4245_4E43);
    let num_runs = 40usize;
    let num_modules = 5usize;
    let tests_per_module = 6usize;

    let mut runs = Vec::with_capacity(num_runs);
    let mut commits = Vec::with_capacity(num_runs);

    for r in 0..num_runs {
        let ts = 1_720_000_000 + (r as i64) * 1800;
        let touched = (rng.next_u64() as usize) % num_modules;
        let diff_files = vec![DiffFile {
            path: format!("src/mod_{touched}.py"),
            lines_added: 20,
            lines_deleted: 6,
        }];
        let diff_fp = DiffSummary::compute_fingerprint(&diff_files);
        commits.push(CommitRecord {
            commit_sha: format!("c-{r:04}"),
            parent_sha: (r > 0).then(|| format!("c-{:04}", r - 1)),
            author_hash: "bench".to_string(),
            commit_ts: ts - 60,
            files_changed: 1,
            lines_added: 20,
            lines_deleted: 6,
            files: diff_files.clone(),
        });

        let mut outcomes = Vec::with_capacity(num_modules * tests_per_module);
        for m in 0..num_modules {
            let file_path = format!("tests/test_mod_{m}.py");
            let hop = if m == touched { Some(1) } else { None };
            for t in 0..tests_per_module {
                let fail_prob = if m == touched && t == 0 {
                    0.80
                } else if m == touched && t == 1 {
                    0.35
                } else {
                    0.02
                };
                let status = if rng.next_f64() < fail_prob {
                    TestStatus::Fail
                } else {
                    TestStatus::Pass
                };
                outcomes.push(HistoricalOutcome {
                    test_id: format!("{file_path}::test_{t}"),
                    file_path: file_path.clone(),
                    status,
                    duration_ms: 80 + ((m * 11 + t * 17) as u64),
                    sampling_propensity: 1.0,
                    ast_min_hop: hop,
                });
            }
        }

        runs.push(HistoricalRun {
            run_id: format!("run-{r:04}"),
            commit_sha: Some(format!("c-{r:04}")),
            diff_fingerprint: diff_fp,
            started_at: ts,
            diff_files,
            outcomes,
        });
    }

    (runs, commits)
}

fn build_feature_batch(batch_size: usize) -> Vec<[f64; NUM_FEATURES]> {
    let mut rng = SplitMix64::new(0xBEEF_CAFE_1234_5678);
    let mut rows = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let mut row = [0.0_f64; NUM_FEATURES];
        row[0] = if i % 17 == 0 { 1.0 } else { 0.0 };
        row[1] = rng.next_f64();
        row[2] = (rng.next_u64() % 4) as f64;
        row[3] = if i % 5 == 0 {
            ((i % 3) + 1) as f64
        } else {
            f64::NAN
        };
        row[4] = rng.next_f64() * 0.4;
        row[5] = rng.next_f64() * 0.2;
        row[6] = rng.next_f64() * 0.3;
        row[7] = rng.next_f64() * 0.25;
        row[8] = (rng.next_u64() % 20) as f64;
        row[9] = (rng.next_u64() % 30) as f64;
        row[10] = (rng.next_u64() % 10) as f64;
        row[11] = (rng.next_u64() % 200) as f64;
        row[12] = rng.next_f64() * 0.05;
        row[13] = 20.0 + rng.next_f64() * 400.0;
        row[14] = (rng.next_u64() % 40) as f64;
        row[15] = if i % 25 == 0 { 1.0 } else { 0.0 };
        rows.push(row);
    }
    rows
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let batch_size = 1_000usize;
    let warmup_iters = 10usize;
    let measured_iters = 200usize;

    let (runs, commits) = build_synthetic_training_history();
    let train_res = train_gbdt_model(&runs, &commits, 1_730_000_000)
        .expect("train synthetic GBDT+Platt model for inference benchmark");
    let model = train_res.model;
    let rows = build_feature_batch(batch_size);

    for _ in 0..warmup_iters {
        black_box(model.predict_probs(black_box(&rows)));
    }

    let mut samples_ms = Vec::with_capacity(measured_iters);
    for _ in 0..measured_iters {
        let start = Instant::now();
        let probs = model.predict_probs(black_box(&rows));
        black_box(&probs);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    samples_ms.sort_by(f64::total_cmp);
    let median_ms = percentile(&samples_ms, 0.50);
    let p95_ms = percentile(&samples_ms, 0.95);
    let mean_ms = samples_ms.iter().sum::<f64>() / (samples_ms.len() as f64);
    let tests_per_sec = if median_ms > 0.0 {
        (batch_size as f64) / (median_ms / 1000.0)
    } else {
        0.0
    };

    let output = serde_json::json!({
        "batch_size": batch_size,
        "iterations": measured_iters,
        "median_ms_per_1000": (median_ms * 1000.0).round() / 1000.0,
        "p95_ms_per_1000": (p95_ms * 1000.0).round() / 1000.0,
        "mean_ms_per_1000": (mean_ms * 1000.0).round() / 1000.0,
        "tests_per_sec": tests_per_sec.round(),
    });
    println!("{output}");
}
