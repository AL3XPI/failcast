//! Chronological walk-forward evaluator and comparison baselines.

pub mod baselines;
pub mod metrics;
pub mod walk_forward;

pub use baselines::{
    RankerKind, score_random_batch, score_recent_failure_rate, score_static_ast, select_for_budget,
};
pub use metrics::{
    CalibrationBin, CalibrationBins, MetricsAccumulator, NUM_CALIBRATION_BINS, StrategyMetrics,
    compute_brier_score, compute_calibration_bins, compute_pr_auc,
};
pub use walk_forward::{BudgetEvalReport, WalkForwardConfig, WalkForwardReport, walk_forward_eval};

#[cfg(test)]
pub mod tests {
    use pts_core::{
        CommitRecord, Config, DiffFile, DiffSummary, HistoricalOutcome, HistoricalRun, TestStatus,
    };

    use super::*;

    /// Deterministic PRNG for synthetic benchmark history generation.
    struct TestRng(u64);
    impl TestRng {
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

    #[test]
    fn walk_forward_beats_random_on_synthetic_history() {
        let mut rng = TestRng::new(20250101);
        let num_runs = 60usize;
        let num_modules = 5usize;
        let tests_per_module = 4usize;

        let mut runs = Vec::with_capacity(num_runs);
        let mut commits = Vec::with_capacity(num_runs);

        for r in 0..num_runs {
            let ts = 1_700_000_000 + (r as i64) * 3600;
            let touched_mod = (rng.next_u64() as usize) % num_modules;
            let is_regression = r % 3 != 0;
            let diff_file = format!("src/mod_{touched_mod}.py");
            let mut diff_files = vec![DiffFile {
                path: diff_file.clone(),
                lines_added: if is_regression { 28 } else { 3 },
                lines_deleted: if is_regression { 10 } else { 1 },
            }];
            if r % 4 == 0 {
                diff_files.push(DiffFile {
                    path: format!("tests/test_mod_{touched_mod}.py"),
                    lines_added: 6,
                    lines_deleted: 1,
                });
            }
            let diff_fp = DiffSummary::compute_fingerprint(&diff_files);

            commits.push(CommitRecord {
                commit_sha: format!("commit-{r:04}"),
                parent_sha: if r == 0 {
                    None
                } else {
                    Some(format!("commit-{:04}", r - 1))
                },
                author_hash: "author-1".to_string(),
                commit_ts: ts - 60,
                files_changed: diff_files.len() as u32,
                lines_added: diff_files.iter().map(|f| f.lines_added).sum(),
                lines_deleted: diff_files.iter().map(|f| f.lines_deleted).sum(),
                files: diff_files.clone(),
            });

            let mut outcomes = Vec::with_capacity(num_modules * tests_per_module);
            for m in 0..num_modules {
                let file_path = format!("tests/test_mod_{m}.py");
                let direct_mod = diff_files.iter().any(|f| f.path == file_path);
                let ast_min_hop = if direct_mod {
                    Some(0)
                } else if m == touched_mod {
                    Some(1)
                } else {
                    None
                };

                for t in 0..tests_per_module {
                    let test_id = format!("{file_path}::test_{t}");
                    let mut fail_prob = 0.01;
                    if is_regression {
                        if m == touched_mod && t == 0 {
                            fail_prob = 0.90;
                        } else if m == touched_mod && t == 1 {
                            fail_prob = 0.45;
                        }
                        if touched_mod == 4 && m == 0 && t == 2 {
                            fail_prob = 0.85;
                        }
                    }

                    let status = if rng.next_f64() < fail_prob {
                        TestStatus::Fail
                    } else {
                        TestStatus::Pass
                    };
                    let duration_ms = 150 + ((m * 17 + t * 23) as u64 % 100);

                    outcomes.push(HistoricalOutcome {
                        test_id,
                        file_path: file_path.clone(),
                        status,
                        duration_ms,
                        sampling_propensity: 1.0,
                        ast_min_hop,
                    });
                }
            }

            runs.push(HistoricalRun {
                run_id: format!("run-{r:04}"),
                commit_sha: Some(format!("commit-{r:04}")),
                diff_fingerprint: diff_fp,
                started_at: ts,
                diff_files,
                outcomes,
            });
        }

        let wf_config = WalkForwardConfig {
            min_train_runs: 20,
            retrain_every: 5,
            budget_fractions: vec![0.1, 0.2, 0.3],
            selector_config: Config {
                explore_fraction: 0.10,
                ..Config::default()
            },
        };

        let report = walk_forward_eval(&runs, &commits, &wf_config).expect("walk_forward_eval");
        assert!(
            report.zero_leakage_verified,
            "zero timestamp leakage must hold"
        );
        assert_eq!(report.total_runs, 60);
        assert_eq!(report.evaluated_runs, 40);
        assert!(
            report.gbdt_steps > 0,
            "GBDT must be active during walk-forward eval"
        );
        assert!(
            report.failing_runs >= 10,
            "expected substantial failing runs"
        );

        let pts_20 = report
            .find_metrics(0.2, RankerKind::PtsGbdt.display_name())
            .expect("PTS metrics at 20% budget");
        let random_20 = report
            .find_metrics(0.2, RankerKind::Random.display_name())
            .expect("Random metrics at 20% budget");

        assert!(
            pts_20.commit_fdr > random_20.commit_fdr,
            "PTS (GBDT+Platt) Commit-Level FDR ({:.3}) must strictly beat Random ({:.3}) at 20% budget",
            pts_20.commit_fdr,
            random_20.commit_fdr
        );
        assert!(
            pts_20.wall_clock_time_saved > 0.60,
            "expected significant wall-clock time savings at 20% budget, got {:.3}",
            pts_20.wall_clock_time_saved
        );
        assert!(
            pts_20.pr_auc > random_20.pr_auc,
            "PTS PR-AUC ({:.3}) should beat Random ({:.3})",
            pts_20.pr_auc,
            random_20.pr_auc
        );

        let md = report.render_markdown();
        assert!(md.contains("PTS (GBDT+Platt)"));
        assert!(md.contains("Random"));
        assert!(md.contains("StaticAst"));
        assert!(md.contains("RecentFailureRate"));
        assert!(md.contains("Heuristic"));

        let json_str = report.to_json_pretty().expect("serialize report JSON");
        let parsed: WalkForwardReport =
            serde_json::from_str(&json_str).expect("deserialize report JSON");
        assert_eq!(parsed, report);
    }

    #[test]
    fn calibration_bins_sum_to_evaluated_outcomes_and_stay_chronological() {
        let mut rng = TestRng::new(20260505);
        let num_runs = 45usize;
        let tests_per_run = 10usize;
        let mut runs = Vec::with_capacity(num_runs);
        let mut commits = Vec::with_capacity(num_runs);

        for r in 0..num_runs {
            let ts = 1_710_000_000 + (r as i64) * 1800;
            let touched = r % 5;
            let diff_files = vec![DiffFile {
                path: format!("src/pkg_{touched}.py"),
                lines_added: 12,
                lines_deleted: 4,
            }];
            let diff_fp = DiffSummary::compute_fingerprint(&diff_files);
            commits.push(CommitRecord {
                commit_sha: format!("c-{r:04}"),
                parent_sha: (r > 0).then(|| format!("c-{:04}", r - 1)),
                author_hash: "dev-1".to_string(),
                commit_ts: ts - 30,
                files_changed: 1,
                lines_added: 12,
                lines_deleted: 4,
                files: diff_files.clone(),
            });

            let mut outcomes = Vec::with_capacity(tests_per_run);
            for t in 0..tests_per_run {
                let module_idx = t % 5;
                let file_path = format!("tests/test_pkg_{module_idx}.py");
                let status = if t == 9 {
                    TestStatus::Skip
                } else if (module_idx == touched && t < 5 && rng.next_f64() < 0.75)
                    || rng.next_f64() < 0.03
                {
                    TestStatus::Fail
                } else {
                    TestStatus::Pass
                };
                outcomes.push(HistoricalOutcome {
                    test_id: format!("{file_path}::test_{t}"),
                    file_path,
                    status,
                    duration_ms: 100 + (t as u64) * 10,
                    sampling_propensity: 1.0,
                    ast_min_hop: if module_idx == touched { Some(1) } else { None },
                });
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

        let wf_config = WalkForwardConfig {
            min_train_runs: 15,
            retrain_every: 5,
            budget_fractions: vec![0.2],
            selector_config: Config::default(),
        };

        let mut reversed_runs = runs.clone();
        reversed_runs.reverse();
        let report_full =
            walk_forward_eval(&reversed_runs, &commits, &wf_config).expect("walk_forward_eval");
        assert!(report_full.zero_leakage_verified);
        assert_eq!(report_full.evaluated_runs, 30);

        let expected_outcomes = 30 * (tests_per_run - 1);
        assert_eq!(
            report_full.calibration_bins.raw_gbdt.len(),
            NUM_CALIBRATION_BINS
        );
        assert_eq!(
            report_full.calibration_bins.platt_calibrated.len(),
            NUM_CALIBRATION_BINS
        );

        let raw_sum: usize = report_full
            .calibration_bins
            .raw_gbdt
            .iter()
            .map(|b| b.count)
            .sum();
        let platt_sum: usize = report_full
            .calibration_bins
            .platt_calibrated
            .iter()
            .map(|b| b.count)
            .sum();
        assert_eq!(raw_sum, expected_outcomes);
        assert_eq!(platt_sum, expected_outcomes);

        for (idx, bin) in report_full
            .calibration_bins
            .platt_calibrated
            .iter()
            .enumerate()
        {
            let expected_center = (idx as f64 + 0.5) / (NUM_CALIBRATION_BINS as f64);
            assert!((bin.bin_center - expected_center).abs() < 1e-12);
            assert!((0.0..=1.0).contains(&bin.mean_predicted));
            assert!((0.0..=1.0).contains(&bin.empirical_positive_rate));
        }

        let pts_metrics = report_full
            .find_metrics(0.2, RankerKind::PtsGbdt.display_name())
            .expect("PTS metrics");
        assert!((report_full.brier_score - pts_metrics.brier_score).abs() < 1e-12);
        assert_eq!(
            pts_metrics.uncalibrated_brier_score,
            Some(report_full.uncalibrated_brier_score)
        );
        assert!(report_full.uncalibrated_brier_score.is_finite());
        assert!(report_full.uncalibrated_brier_score >= 0.0);

        // Strict chronological invariance: evaluating a prefix of 30 runs produces the exact same
        // calibration bins regardless of whether future runs (30..45) exist or have flipped labels.
        let prefix_report =
            walk_forward_eval(&runs[..30], &commits, &wf_config).expect("prefix eval");
        let mut mutated_prefix_with_future = runs.clone();
        for run in &mut mutated_prefix_with_future[30..] {
            for outcome in &mut run.outcomes {
                if outcome.status != TestStatus::Skip {
                    outcome.status = TestStatus::Fail;
                }
            }
        }
        let prefix_after_mutation =
            walk_forward_eval(&mutated_prefix_with_future[..30], &commits, &wf_config)
                .expect("prefix eval after future mutation");
        assert_eq!(
            prefix_report.calibration_bins,
            prefix_after_mutation.calibration_bins
        );
        assert!(
            (prefix_report.uncalibrated_brier_score
                - prefix_after_mutation.uncalibrated_brier_score)
                .abs()
                < 1e-12
        );

        // Backwards compatibility: JSON omitting the newly added fields still deserializes cleanly.
        let mut legacy_json: serde_json::Value =
            serde_json::to_value(&report_full).expect("serialize to Value");
        let obj = legacy_json.as_object_mut().expect("root JSON object");
        obj.remove("brier_score");
        obj.remove("uncalibrated_brier_score");
        obj.remove("calibration_bins");
        let parsed_legacy: WalkForwardReport =
            serde_json::from_value(legacy_json).expect("deserialize legacy JSON");
        assert_eq!(parsed_legacy.evaluated_runs, report_full.evaluated_runs);
        assert!(parsed_legacy.calibration_bins.raw_gbdt.is_empty());
    }
}
