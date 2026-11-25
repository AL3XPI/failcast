//! Pure feature engineering, cold-start heuristic ranker, GBDT trainer, Platt calibrator,
//! and cost-aware selector.
//!
//! Invariant 2: This crate is 100% pure — no disk I/O, no subprocesses, no DB,
//! and no `SystemTime::now()` calls.

pub mod accumulator;
pub mod features;
pub mod gbdt;
pub mod heuristic;
pub mod platt;
pub mod selector;

pub use accumulator::{HistoryAccumulator, TrainingRow};
pub use features::{FEATURE_NAMES, NUM_FEATURES, extract_features};
pub use gbdt::{
    BOOSTER_PREDICTS_RAW_MARGINS, ChronologicalSplit, GBDT_ITERATIONS, GBDT_L2, GBDT_LEARNING_RATE,
    GBDT_MAX_DEPTH, GBDT_MIN_LEAF_WEIGHT, GBDT_SEED, MODEL_ARTIFACT_VERSION, ModelArtifact,
    TrainResult, TrainedModel, build_chronological_split, build_default_booster,
    inverse_propensity_weight, normalize_to_log_odds, to_column_major, train_gbdt_model,
};
pub use heuristic::{ast_hop_decay, score_heuristic, should_use_heuristic};
pub use platt::{
    MIN_CAL_FAILING_RUNS, MIN_CAL_POSITIVES, PlattCalibrator, brier_score, logit, logit_log_loss,
    sigmoid,
};
pub use selector::{
    CandidateTest, DEFAULT_MEDIAN_DURATION_MS, MIN_EFF_DURATION_MS, SelectedTest,
    SelectionStrategy, select_tests,
};

#[cfg(test)]
pub mod tests {
    use pts_core::{
        CoFailCounts, Config, DiffFile, DiffSummary, HistorySnapshot, SelectionReason, TestCase,
        TestHistory,
    };

    use super::*;
    use crate::features::*;

    #[test]
    fn features_are_pure_and_handle_missing_values() {
        let diff = DiffSummary::from_files(
            Some("abc123".to_string()),
            vec![
                DiffFile {
                    path: "src/auth/token.py".to_string(),
                    lines_added: 12,
                    lines_deleted: 3,
                },
                DiffFile {
                    path: "tests/test_other.py".to_string(),
                    lines_added: 4,
                    lines_deleted: 1,
                },
            ],
        );

        let mut history = HistorySnapshot {
            as_of_ts: 1_700_000_000,
            total_runs: 10,
            total_failures: 3,
            median_repo_test_ms: Some(150.0),
            ..HistorySnapshot::default()
        };
        history.test_history.insert(
            "tests/test_auth.py::test_refresh".to_string(),
            TestHistory {
                test_id: "tests/test_auth.py::test_refresh".to_string(),
                total_runs: 8,
                total_failures: 2,
                ewma_fail_10: 0.32,
                ewma_fail_50: 0.11,
                runs_since_last_fail: Some(2),
                median_duration_ms: Some(180.0),
                flakiness_rate: 0.25,
                last_run_ts: Some(1_699_999_000),
            },
        );
        history.cofail_counts.insert(
            (
                "tests/test_auth.py::test_refresh".to_string(),
                "src/auth/token.py".to_string(),
            ),
            CoFailCounts {
                co_fails: 3,
                co_runs: 5,
            },
        );
        history
            .file_commit_counts
            .insert("src/auth/token.py".to_string(), 4);
        history.pair_commit_counts.insert(
            (
                "src/auth/token.py".to_string(),
                "tests/test_auth.py".to_string(),
            ),
            3,
        );
        history
            .file_churn_30d
            .insert("src/auth/token.py".to_string(), 7);
        history
            .file_churn_30d
            .insert("tests/test_other.py".to_string(), 2);

        // Case 1: Seen test with reachable AST hop = 1.
        let seen_test = TestCase {
            test_id: "tests/test_auth.py::test_refresh".to_string(),
            file_path: "tests/test_auth.py".to_string(),
        };
        let f_seen = extract_features(&seen_test, &diff, &history, Some(1));
        assert_eq!(f_seen.len(), NUM_FEATURES);
        assert!((f_seen[IDX_AST_MIN_HOP] - 1.0).abs() < 1e-12);
        assert!((f_seen[IDX_DIRECT_TEST_MODIFIED] - 0.0).abs() < 1e-12);
        assert!((f_seen[IDX_PATH_JACCARD_MAX] - 0.5).abs() < 1e-12);
        assert!((f_seen[IDX_SAME_DIR_OVERLAP] - 1.0).abs() < 1e-12);
        // Laplace smoothed: (3 + 0.5) / (5 + 5.0) = 0.35
        assert!((f_seen[IDX_COFAIL_RATE_MAX] - 0.35).abs() < 1e-12);
        // Co-change: 3 / 4 = 0.75
        assert!((f_seen[IDX_COCHANGE_FREQ_MAX] - 0.75).abs() < 1e-12);
        assert!((f_seen[IDX_EWMA_FAIL_10] - 0.32).abs() < 1e-12);
        assert!((f_seen[IDX_EWMA_FAIL_50] - 0.11).abs() < 1e-12);
        assert!((f_seen[IDX_RUNS_SINCE_LAST_FAIL] - 2.0).abs() < 1e-12);
        assert!((f_seen[IDX_TOTAL_TEST_RUNS] - 8.0).abs() < 1e-12);
        assert!((f_seen[IDX_MEDIAN_DURATION_MS] - 180.0).abs() < 1e-12);
        assert!((f_seen[IDX_FLAKINESS_RATE] - 0.25).abs() < 1e-12);
        assert!((f_seen[IDX_DIFF_FILES_COUNT] - 2.0).abs() < 1e-12);
        assert!((f_seen[IDX_DIFF_LINES_TOTAL] - (20.0_f64).ln_1p()).abs() < 1e-12);
        assert!((f_seen[IDX_DIFF_TOP_DIRS_COUNT] - 2.0).abs() < 1e-12);
        assert!((f_seen[IDX_TOUCHED_FILE_CHURN_30D] - 7.0).abs() < 1e-12);

        // Case 2: Unseen test with unreachable AST hop (None) -> NAN in indices 0, 8, 10.
        let unseen_test = TestCase {
            test_id: "unit/test_brand_new.py::test_new".to_string(),
            file_path: "unit/test_brand_new.py".to_string(),
        };
        let f_unseen = extract_features(&unseen_test, &diff, &history, None);
        assert!(
            f_unseen[IDX_AST_MIN_HOP].is_nan(),
            "unreachable AST hop must be NAN"
        );
        assert!((f_unseen[IDX_DIRECT_TEST_MODIFIED] - 0.0).abs() < 1e-12);
        assert!((f_unseen[IDX_PATH_JACCARD_MAX] - 0.0).abs() < 1e-12);
        assert!((f_unseen[IDX_SAME_DIR_OVERLAP] - 0.0).abs() < 1e-12);
        // Unseen co-failure Laplace prior: (0 + 0.5) / (0 + 5.0) = 0.1
        assert!((f_unseen[IDX_COFAIL_RATE_MAX] - 0.1).abs() < 1e-12);
        assert!((f_unseen[IDX_COCHANGE_FREQ_MAX] - 0.0).abs() < 1e-12);
        assert!((f_unseen[IDX_EWMA_FAIL_10] - 0.0).abs() < 1e-12);
        assert!((f_unseen[IDX_EWMA_FAIL_50] - 0.0).abs() < 1e-12);
        assert!(
            f_unseen[IDX_RUNS_SINCE_LAST_FAIL].is_nan(),
            "never-failed test must have NAN runs_since_last_fail"
        );
        assert!((f_unseen[IDX_TOTAL_TEST_RUNS] - 0.0).abs() < 1e-12);
        assert!(
            f_unseen[IDX_MEDIAN_DURATION_MS].is_nan(),
            "unseen test must have NAN median_duration_ms"
        );
        assert!((f_unseen[IDX_FLAKINESS_RATE] - 0.0).abs() < 1e-12);
        assert!((f_unseen[IDX_DIFF_FILES_COUNT] - 2.0).abs() < 1e-12);
        assert!((f_unseen[IDX_DIFF_LINES_TOTAL] - (20.0_f64).ln_1p()).abs() < 1e-12);
        assert!((f_unseen[IDX_DIFF_TOP_DIRS_COUNT] - 2.0).abs() < 1e-12);
        assert!((f_unseen[IDX_TOUCHED_FILE_CHURN_30D] - 7.0).abs() < 1e-12);

        // Verify heuristic ranker produces strictly higher score for directly-related seen test.
        let p_seen = score_heuristic(&f_seen);
        let p_unseen = score_heuristic(&f_unseen);
        assert!(p_seen > 0.0 && p_seen < 1.0);
        assert!(p_unseen > 0.0 && p_unseen < 1.0);
        assert!(p_seen > p_unseen);
    }

    #[test]
    fn selector_respects_budget_explores_and_is_deterministic() {
        let config = Config {
            explore_fraction: 0.25, // 250ms explore, 750ms exploit out of 1000ms budget
            ..Config::default()
        };

        // `test_fast_high_yield`: prob=0.70, dur=100ms -> yield = 0.0070
        // `test_slow_higher_prob`: prob=0.80, dur=800ms -> yield = 0.0010 (won't fit exploit after fast)
        // `test_medium`: prob=0.50, dur=500ms -> yield = 0.0010 (fits exploit: 100 + 500 = 600 <= 750)
        // `test_stale_a`, `test_stale_b`: dur=100ms, low prob -> eligible for explore (250ms budget)
        let candidates = vec![
            CandidateTest {
                test_id: "tests/test_suite.py::test_slow_higher_prob".to_string(),
                prob: 0.80,
                est_duration_ms: Some(800.0),
                days_since_last_run: 1.0,
            },
            CandidateTest {
                test_id: "tests/test_suite.py::test_fast_high_yield".to_string(),
                prob: 0.70,
                est_duration_ms: Some(100.0),
                days_since_last_run: 0.5,
            },
            CandidateTest {
                test_id: "tests/test_suite.py::test_medium".to_string(),
                prob: 0.50,
                est_duration_ms: Some(500.0),
                days_since_last_run: 2.0,
            },
            CandidateTest {
                test_id: "tests/test_suite.py::test_stale_a".to_string(),
                prob: 0.05,
                est_duration_ms: Some(100.0),
                days_since_last_run: 14.0,
            },
            CandidateTest {
                test_id: "tests/test_suite.py::test_stale_b".to_string(),
                prob: 0.04,
                est_duration_ms: Some(100.0),
                days_since_last_run: 30.0,
            },
            CandidateTest {
                test_id: "tests/test_suite.py::test_stale_c".to_string(),
                prob: 0.03,
                est_duration_ms: Some(100.0),
                days_since_last_run: 10.0,
            },
        ];

        let fingerprint = "a1b2c3d4e5f6071899aabbccddeeff00112233445566778899aabbccddeeff00";
        let strategy = SelectionStrategy::Budget { budget_ms: 1000 };

        let run1 = select_tests(
            &candidates,
            strategy,
            fingerprint,
            Some(200.0),
            &config,
            SelectionReason::Heuristic,
        );

        // Repeat with reversed input order to verify full determinism regardless of input ordering.
        let mut reversed = candidates.clone();
        reversed.reverse();
        let run2 = select_tests(
            &reversed,
            strategy,
            fingerprint,
            Some(200.0),
            &config,
            SelectionReason::Heuristic,
        );

        assert_eq!(
            serde_json::to_string(&run1).expect("serialize run1"),
            serde_json::to_string(&run2).expect("serialize run2"),
            "selection output must be byte-for-byte deterministic"
        );

        // First selected test MUST be the high-yield fast test, NOT the slow slightly-higher-prob test.
        assert_eq!(run1[0].test_id, "tests/test_suite.py::test_fast_high_yield");
        assert_eq!(run1[0].selection_reason, SelectionReason::Heuristic);
        assert!((run1[0].sampling_propensity - 1.0).abs() < 1e-12);

        // Slow 800ms test must NOT be selected in exploit (exceeds remaining exploit budget).
        assert!(
            !run1
                .iter()
                .any(|s| s.test_id == "tests/test_suite.py::test_slow_higher_prob")
        );

        // Total estimated duration must stay within 1000ms budget.
        let total_ms: u64 = run1.iter().map(|s| s.est_duration_ms).sum();
        assert!(
            total_ms <= 1000,
            "total est duration {total_ms}ms exceeded 1000ms budget"
        );

        // Explore pass must allocate tests with SelectionReason::Explore and valid sampling_propensity.
        let explored: Vec<&SelectedTest> = run1
            .iter()
            .filter(|s| s.selection_reason == SelectionReason::Explore)
            .collect();
        assert!(
            !explored.is_empty(),
            "explore pass must select stale candidate(s)"
        );
        for item in explored {
            assert!(
                item.sampling_propensity >= 0.01 && item.sampling_propensity <= 1.0,
                "propensity {} out of bounds",
                item.sampling_propensity
            );
            // With 3 eligible stale candidates on first draw, propensity must be < 1.0.
            assert!(item.sampling_propensity < 1.0 || run1.len() >= 4);
        }
    }

    #[test]
    fn selector_supports_top_k_target_recall_and_lexicographic_ties() {
        let config = Config::default();
        let candidates = vec![
            CandidateTest {
                test_id: "tests/test_b.py::test_2".to_string(),
                prob: 0.5,
                est_duration_ms: Some(10.0), // floored to 50ms
                days_since_last_run: 0.0,
            },
            CandidateTest {
                test_id: "tests/test_a.py::test_1".to_string(),
                prob: 0.5,
                est_duration_ms: Some(10.0), // floored to 50ms -> exact yield tie with test_b
                days_since_last_run: 0.0,
            },
            CandidateTest {
                test_id: "tests/test_c.py::test_3".to_string(),
                prob: 0.05,
                est_duration_ms: Some(100.0),
                days_since_last_run: 0.0,
            },
        ];

        // Top-K breaks ties lexicographically by test_id.
        let top1 = select_tests(
            &candidates,
            SelectionStrategy::TopK { k: 1 },
            "0000000000000000",
            None,
            &config,
            SelectionReason::Ranked,
        );
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].test_id, "tests/test_a.py::test_1");
        assert_eq!(top1[0].est_duration_ms, 50);

        // Target-Recall 0.90:
        // unselected [0.5, 0.5, 0.05] -> miss=0.7625 > 0.10 (pick test_a)
        // unselected [0.5, 0.05] -> miss=0.525 > 0.10 (pick test_b)
        // unselected [0.05] -> miss=0.05 <= 0.10 (stop!)
        let recall_sel = select_tests(
            &candidates,
            SelectionStrategy::TargetRecall {
                target_recall: 0.90,
                max_budget_ms: None,
            },
            "0000000000000000",
            None,
            &config,
            SelectionReason::Ranked,
        );
        assert_eq!(recall_sel.len(), 2);
        assert_eq!(recall_sel[0].test_id, "tests/test_a.py::test_1");
        assert_eq!(recall_sel[1].test_id, "tests/test_b.py::test_2");
    }

    fn make_synthetic_runs(num_runs: usize) -> Vec<pts_core::HistoricalRun> {
        use pts_core::{DiffFile, HistoricalOutcome, HistoricalRun, TestStatus};

        let mut runs = Vec::with_capacity(num_runs);
        for r in 0..num_runs {
            let started_at = 1_700_000_000 + (r as i64) * 3600;
            let mod_idx = r % 3;
            let diff_file = format!("src/mod_{mod_idx}.py");
            let is_bug = r % 2 == 1;
            let diff_files = vec![DiffFile {
                path: diff_file.clone(),
                lines_added: if is_bug { 25 } else { 4 },
                lines_deleted: if is_bug { 8 } else { 1 },
            }];
            let fp = DiffSummary::compute_fingerprint(&diff_files);

            let mut outcomes = Vec::new();
            for t in 0..6 {
                let test_mod = t / 2;
                let file_path = format!("tests/test_mod_{test_mod}.py");
                let test_id = format!("{file_path}::test_{t}");
                let coupled = test_mod == mod_idx;
                let status = if t == 5 && r % 5 == 0 {
                    TestStatus::Skip
                } else if is_bug && coupled && t % 2 == 0 {
                    TestStatus::Fail
                } else {
                    TestStatus::Pass
                };
                outcomes.push(HistoricalOutcome {
                    test_id,
                    file_path,
                    status,
                    duration_ms: 100 + (t as u64) * 25,
                    sampling_propensity: if t % 3 == 0 { 0.5 } else { 1.0 },
                    ast_min_hop: if coupled { Some(1) } else { None },
                });
            }

            runs.push(HistoricalRun {
                run_id: format!("run-{r:04}"),
                commit_sha: Some(format!("sha-{r:04}")),
                diff_fingerprint: fp,
                started_at,
                diff_files,
                outcomes,
            });
        }
        runs
    }

    #[test]
    fn column_major_layout_and_booster_determinism() {
        // 1. Verify to_column_major exact indexing: out[col * num_rows + row] == rows[row][col].
        let mut row0 = [0.0; NUM_FEATURES];
        let mut row1 = [0.0; NUM_FEATURES];
        let mut row2 = [0.0; NUM_FEATURES];
        for c in 0..NUM_FEATURES {
            row0[c] = (c as f64) + 0.1;
            row1[c] = (c as f64) + 100.2;
            row2[c] = if c == 0 { f64::NAN } else { (c as f64) + 200.3 };
        }
        let rows = [row0, row1, row2];
        let col_major = to_column_major(&rows);
        assert_eq!(col_major.len(), rows.len() * NUM_FEATURES);
        for (r_idx, row) in rows.iter().enumerate() {
            for (c_idx, &expected) in row.iter().enumerate() {
                let actual = col_major[c_idx * rows.len() + r_idx];
                if expected.is_nan() {
                    assert!(actual.is_nan(), "expected NaN at ({r_idx}, {c_idx})");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "mismatch at ({r_idx}, {c_idx})"
                    );
                }
            }
        }

        // 2. Verify forust-ml LogLoss booster outputs raw log-odds margins (values < 0.0 and > 1.0).
        let runs = make_synthetic_runs(30);
        let res1 = train_gbdt_model(&runs, &[], 1_700_500_000).expect("first training");
        let res2 = train_gbdt_model(&runs, &[], 1_700_500_000).expect("second training");

        let probe_rows: Vec<[f64; NUM_FEATURES]> = HistoryAccumulator::build_dataset(&runs, &[])
            .into_iter()
            .map(|r| r.features)
            .collect();

        let raw_margins = res1.model.predict_raw_margins(&probe_rows);
        assert!(
            raw_margins.iter().any(|&z| z < 0.0),
            "expected negative raw log-odds margins from LogLoss booster, got range [{:?}, {:?}]",
            raw_margins.iter().copied().reduce(f64::min),
            raw_margins.iter().copied().reduce(f64::max),
        );
        assert!(
            raw_margins.iter().any(|&z| z > 1.0) || raw_margins.iter().any(|&z| z < -1.0),
            "expected raw log-odds margins outside [0, 1] probability range"
        );

        // 3. Confirm bit-identical predictions and serialized artifacts across two trainings.
        let probs1 = res1.model.predict_probs(&probe_rows);
        let probs2 = res2.model.predict_probs(&probe_rows);
        assert_eq!(probs1.len(), probs2.len());
        for (i, (&p1, &p2)) in probs1.iter().zip(probs2.iter()).enumerate() {
            assert_eq!(
                p1.to_bits(),
                p2.to_bits(),
                "prediction mismatch at row {i}: {p1} vs {p2}"
            );
            assert!(p1 > 0.0 && p1 < 1.0);
        }

        let json1 = res1.model.to_json().expect("serialize artifact 1");
        let json2 = res2.model.to_json().expect("serialize artifact 2");
        assert_eq!(
            json1, json2,
            "serialized ModelArtifact must be bit-identical"
        );

        // 4. Confirm deserialized model reproduces bit-identical predictions.
        let reloaded = TrainedModel::from_json(&json1).expect("deserialize artifact");
        let probs_reloaded = reloaded.predict_probs(&probe_rows);
        for (&p1, &pr) in probs1.iter().zip(probs_reloaded.iter()) {
            assert_eq!(p1.to_bits(), pr.to_bits());
        }
    }

    #[test]
    fn platt_calibrator_improves_brier_and_handles_degenerate_splits() {
        // 1. Construct uncalibrated, overconfident, shifted synthetic logits:
        //    true p = sigmoid(x), but raw model outputs z = 3.0 * x + 2.0.
        let mut raw_logits = Vec::new();
        let mut labels = Vec::new();
        for i in 0..200 {
            let x = -3.0 + 6.0 * (i as f64) / 199.0;
            let true_p = sigmoid(x);
            // Deterministic stratified pseudo-sampling matching true_p.
            let threshold = ((i * 37 + 13) % 100) as f64 / 100.0;
            let y = threshold < true_p;
            raw_logits.push(3.0 * x + 2.0);
            labels.push(y);
        }

        let raw_probs: Vec<f64> = raw_logits.iter().map(|&z| sigmoid(z)).collect();
        let raw_brier = brier_score(&raw_probs, &labels);

        let cal = PlattCalibrator::fit(&raw_logits, &labels, 0.5);
        assert!(cal.a.is_finite() && cal.b.is_finite());
        assert!(
            cal.a >= 0.0,
            "Platt slope a must be non-negative, got {}",
            cal.a
        );

        let cal_probs: Vec<f64> = raw_logits.iter().map(|&z| cal.predict_prob(z)).collect();
        let cal_brier = brier_score(&cal_probs, &labels);
        assert!(
            cal_brier < raw_brier,
            "Platt calibration must lower Brier score: cal={cal_brier:.6} vs raw={raw_brier:.6}"
        );

        // 2. Inverted logits (where unconstrained slope would be negative) must fall back to identity (a=1, b=0).
        let inverted_logits: Vec<f64> = vec![-3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let inverted_labels: Vec<bool> = vec![
            true, true, true, true, true, false, false, false, false, false,
        ];
        let inv_cal = PlattCalibrator::fit(&inverted_logits, &inverted_labels, 0.25);
        assert!(inv_cal.a > 0.0 && inv_cal.a.is_finite() && inv_cal.b.is_finite());
        assert!((inv_cal.a - 1.0).abs() < 1e-12);
        assert!((inv_cal.b - 0.0).abs() < 1e-12);

        // 3. Degenerate 0-positive and 0-negative calibration splits must fall back deterministically to identity without NaN.
        let zero_pos =
            PlattCalibrator::fit(&[-1.0, 0.0, 1.0, 2.0], &[false, false, false, false], 0.15);
        assert!((zero_pos.a - 1.0).abs() < 1e-12);
        assert!((zero_pos.b - 0.0).abs() < 1e-12);
        assert!(zero_pos.predict_prob(0.5).is_finite());

        let zero_neg = PlattCalibrator::fit(&[-1.0, 0.0, 1.0], &[true, true, true], 0.0);
        assert!((zero_neg.a - 1.0).abs() < 1e-12);
        assert!((zero_neg.b - 0.0).abs() < 1e-12);
        assert!(zero_neg.predict_prob(0.5).is_finite());
    }

    #[test]
    fn platt_expands_cal_window_and_falls_back_to_identity() {
        use pts_core::{DiffFile, DiffSummary, HistoricalOutcome, HistoricalRun, TestStatus};

        // Helper to build N chronological runs where specified run indices have `fail_count` failures.
        let build_runs =
            |total_runs: usize, failing_map: &[(usize, usize)]| -> Vec<HistoricalRun> {
                let mut runs = Vec::with_capacity(total_runs);
                for r in 0..total_runs {
                    let ts = 1_720_000_000 + (r as i64) * 3600;
                    let fail_count = failing_map
                        .iter()
                        .find(|&&(idx, _)| idx == r)
                        .map(|&(_, c)| c)
                        .unwrap_or(0);
                    let diff_files = vec![DiffFile {
                        path: format!("src/mod_{}.py", r % 3),
                        lines_added: 10,
                        lines_deleted: 2,
                    }];
                    let diff_fp = DiffSummary::compute_fingerprint(&diff_files);
                    let mut outcomes = Vec::with_capacity(8);
                    for t in 0..8 {
                        outcomes.push(HistoricalOutcome {
                            test_id: format!("tests/test_m.py::test_{t}"),
                            file_path: "tests/test_m.py".to_string(),
                            status: if t < fail_count {
                                TestStatus::Fail
                            } else {
                                TestStatus::Pass
                            },
                            duration_ms: 50,
                            sampling_propensity: 1.0,
                            ast_min_hop: if t < 2 { Some(1) } else { None },
                        });
                    }
                    runs.push(HistoricalRun {
                        run_id: format!("run-{r:03}"),
                        commit_sha: Some(format!("sha-{r:03}")),
                        diff_fingerprint: diff_fp,
                        started_at: ts,
                        diff_files,
                        outcomes,
                    });
                }
                runs
            };

        // Case 1: 20 runs total -> default 80/20 split is runs[0..16] train, runs[16..20] cal.
        // Put only 1 failing run in tail 20% (run 18, 3 fails) and a 2nd failing run at run 14 (3 fails)
        // plus training failures at runs 2 and 5. The splitter must expand cal_set backward to include
        // run 14 (split_idx = 14, >= 60% = 12) while strictly preserving max_train_ts < min_cal_ts.
        let runs_expand = build_runs(20, &[(2, 3), (5, 3), (14, 3), (18, 3)]);
        let split = build_chronological_split(&runs_expand, &[]).expect("expanded split");
        assert!(
            split.max_train_ts < split.min_cal_ts,
            "strict anti-leakage must hold after window expansion"
        );
        assert_eq!(split.train_run_count, 14);
        assert_eq!(split.cal_run_count, 6);
        assert_eq!(split.cal_failing_runs, 2);
        assert_eq!(split.cal_positives, 6);

        // Case 2: Insufficient support even down to 60% (only 1 failing run in runs[12..20]).
        // Training must succeed and deterministically fall back to identity sigmoid (a=1.0, b=0.0).
        let runs_sparse = build_runs(20, &[(2, 4), (5, 4), (18, 3)]);
        let trained_sparse =
            train_gbdt_model(&runs_sparse, &[], 1_730_000_000).expect("train sparse");
        assert!(
            trained_sparse.calibration_fallback,
            "insufficient cal_set support must set calibration_fallback = true"
        );
        assert!(trained_sparse.model.artifact.calibration_fallback);
        assert!((trained_sparse.model.artifact.platt_a - 1.0).abs() < 1e-12);
        assert!((trained_sparse.model.artifact.platt_b - 0.0).abs() < 1e-12);

        // Case 3: Perfectly calibrated logits where fitted Platt cannot strictly improve log-loss
        // over identity (a=1.0, b=0.0) must trigger identity fallback.
        let mut cal_logits = Vec::new();
        let mut cal_labels = Vec::new();
        for _ in 0..10 {
            cal_logits.push(2.197_224_577); // sigmoid ~= 0.9
            cal_labels.push(true);
        }
        cal_logits.push(2.197_224_577);
        cal_labels.push(false);
        for _ in 0..10 {
            cal_logits.push(-2.197_224_577); // sigmoid ~= 0.1
            cal_labels.push(false);
        }
        cal_logits.push(-2.197_224_577);
        cal_labels.push(true);
        // Check guardrail directly: if Platt doesn't beat identity log-loss, fallback is true;
        // and if distinct_failing_runs < 2, fallback is always true.
        let (cal_low_runs, fb_low_runs) =
            PlattCalibrator::fit_with_guardrail(&cal_logits, &cal_labels, 1);
        assert!(fb_low_runs);
        assert!((cal_low_runs.a - 1.0).abs() < 1e-12);
        assert!((cal_low_runs.b - 0.0).abs() < 1e-12);

        // Inverted logits with sufficient support also fail log-loss & slope guardrail -> identity fallback.
        let inv_logits = [-3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let inv_labels = [
            true, true, true, true, true, false, false, false, false, false,
        ];
        let (cal_inv, fb_inv) = PlattCalibrator::fit_with_guardrail(&inv_logits, &inv_labels, 3);
        assert!(fb_inv);
        assert!((cal_inv.a - 1.0).abs() < 1e-12);
        assert!((cal_inv.b - 0.0).abs() < 1e-12);

        // Case 4: Backwards-compatible ModelArtifact deserialization when calibration_fallback is omitted.
        let mut artifact_val: serde_json::Value =
            serde_json::to_value(&trained_sparse.model.artifact).expect("to_value");
        artifact_val
            .as_object_mut()
            .expect("json obj")
            .remove("calibration_fallback");
        let parsed: ModelArtifact =
            serde_json::from_value(artifact_val).expect("legacy ModelArtifact");
        assert!(!parsed.calibration_fallback);
    }

    #[test]
    fn chronological_split_has_zero_leakage_across_runs() {
        use std::collections::HashSet;

        use pts_core::{CommitRecord, DiffFile, RunMode, TestStatus};
        use pts_store::{RunRecord, Store, TestOutcomeRecord};

        // 1. Verify strict chronological 80/20 split with zero run overlap and skip exclusion.
        let runs = make_synthetic_runs(25);
        let split = build_chronological_split(&runs, &[]).expect("build chronological split");

        assert!(
            split.max_train_ts < split.min_cal_ts,
            "strict anti-leakage violated: max_train_ts={} >= min_cal_ts={}",
            split.max_train_ts,
            split.min_cal_ts
        );

        let train_run_ids: HashSet<&str> =
            split.train_rows.iter().map(|r| r.run_id.as_str()).collect();
        let cal_run_ids: HashSet<&str> = split.cal_rows.iter().map(|r| r.run_id.as_str()).collect();
        assert!(
            train_run_ids.is_disjoint(&cal_run_ids),
            "no run_id may appear in both train and calibration splits"
        );
        assert!(
            split
                .train_rows
                .iter()
                .all(|r| r.started_at <= split.max_train_ts)
        );
        assert!(
            split
                .cal_rows
                .iter()
                .all(|r| r.started_at >= split.min_cal_ts)
        );
        assert_eq!(train_run_ids.len() + cal_run_ids.len(), runs.len());

        // Verify skipped outcomes were excluded.
        let expected_non_skip: usize = runs
            .iter()
            .flat_map(|r| r.outcomes.iter())
            .filter(|o| o.status != TestStatus::Skip)
            .count();
        assert_eq!(
            split.train_rows.len() + split.cal_rows.len(),
            expected_non_skip
        );

        // 2. Verify HistoryAccumulator matches Store::load_history_snapshot at an intermediate cutoff.
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("store.db");
        let mut store = Store::open(&db_path).expect("open store");
        store.migrate().expect("migrate");

        let commits = vec![
            // Older than 30 days relative to cutoff_ts (should be in cochange, NOT in 30d churn).
            CommitRecord {
                commit_sha: "c_old".to_string(),
                parent_sha: None,
                author_hash: "a1".to_string(),
                commit_ts: 1_690_000_000,
                files_changed: 2,
                lines_added: 10,
                lines_deleted: 2,
                files: vec![
                    DiffFile {
                        path: "src/mod_0.py".to_string(),
                        lines_added: 6,
                        lines_deleted: 1,
                    },
                    DiffFile {
                        path: "tests/test_mod_0.py".to_string(),
                        lines_added: 4,
                        lines_deleted: 1,
                    },
                ],
            },
            // Within 30 days before cutoff_ts.
            CommitRecord {
                commit_sha: "c_recent".to_string(),
                parent_sha: Some("c_old".to_string()),
                author_hash: "a1".to_string(),
                commit_ts: 1_700_005_000,
                files_changed: 2,
                lines_added: 8,
                lines_deleted: 0,
                files: vec![
                    DiffFile {
                        path: "src/mod_0.py".to_string(),
                        lines_added: 5,
                        lines_deleted: 0,
                    },
                    DiffFile {
                        path: "src/mod_1.py".to_string(),
                        lines_added: 3,
                        lines_deleted: 0,
                    },
                ],
            },
            // Strictly after cutoff_ts (must be excluded by both Store and HistoryAccumulator).
            CommitRecord {
                commit_sha: "c_future".to_string(),
                parent_sha: Some("c_recent".to_string()),
                author_hash: "a2".to_string(),
                commit_ts: 1_700_050_000,
                files_changed: 1,
                lines_added: 50,
                lines_deleted: 0,
                files: vec![DiffFile {
                    path: "src/mod_2.py".to_string(),
                    lines_added: 50,
                    lines_deleted: 0,
                }],
            },
        ];
        store.upsert_commits(&commits).expect("upsert commits");

        for run in &runs {
            let run_rec = RunRecord {
                run_id: run.run_id.clone(),
                commit_sha: run.commit_sha.clone(),
                diff_fingerprint: run.diff_fingerprint.clone(),
                runner: "pytest".to_string(),
                mode: RunMode::All,
                budget_ms: None,
                started_at: run.started_at,
                finished_at: Some(run.started_at + 10),
                total_duration_ms: Some(1000),
                exit_code: Some(0),
            };
            let tests: Vec<TestCase> = run
                .outcomes
                .iter()
                .map(|o| TestCase {
                    test_id: o.test_id.clone(),
                    file_path: o.file_path.clone(),
                })
                .collect();
            let outcomes: Vec<TestOutcomeRecord> = run
                .outcomes
                .iter()
                .map(|o| TestOutcomeRecord {
                    run_id: run.run_id.clone(),
                    test_id: o.test_id.clone(),
                    status: o.status,
                    duration_ms: o.duration_ms,
                    predicted_prob: None,
                    selection_reason: SelectionReason::Full,
                    sampling_propensity: o.sampling_propensity,
                    failure_excerpt: None,
                })
                .collect();
            store
                .persist_run_with_outcomes(
                    &run_rec,
                    &run.diff_files,
                    &tests,
                    &outcomes,
                    run.started_at,
                )
                .expect("persist run");
        }

        // Intermediate cutoff: strictly at run[10].started_at -> includes runs 0..10, excludes 10..25.
        let cutoff_ts = runs[10].started_at;
        let store_snap = store
            .load_history_snapshot(cutoff_ts)
            .expect("load_history_snapshot");

        let mut acc = HistoryAccumulator::new(commits);
        for run in &runs[..10] {
            acc.record_run(run);
        }
        let acc_snap = acc.snapshot(cutoff_ts);

        assert_eq!(acc_snap.as_of_ts, store_snap.as_of_ts);
        assert_eq!(acc_snap.total_runs, store_snap.total_runs);
        assert_eq!(acc_snap.total_failures, store_snap.total_failures);
        assert_eq!(acc_snap.median_repo_test_ms, store_snap.median_repo_test_ms);
        assert_eq!(acc_snap.cofail_counts, store_snap.cofail_counts);
        assert_eq!(acc_snap.file_commit_counts, store_snap.file_commit_counts);
        assert_eq!(acc_snap.pair_commit_counts, store_snap.pair_commit_counts);
        assert_eq!(acc_snap.file_churn_30d, store_snap.file_churn_30d);
        assert_eq!(acc_snap.test_history.len(), store_snap.test_history.len());
        for (tid, expected) in &store_snap.test_history {
            let actual = acc_snap.test_history.get(tid).expect("matching test_id");
            assert_eq!(actual.total_runs, expected.total_runs);
            assert_eq!(actual.total_failures, expected.total_failures);
            assert!((actual.ewma_fail_10 - expected.ewma_fail_10).abs() < 1e-12);
            assert!((actual.ewma_fail_50 - expected.ewma_fail_50).abs() < 1e-12);
            assert_eq!(actual.runs_since_last_fail, expected.runs_since_last_fail);
            assert_eq!(actual.median_duration_ms, expected.median_duration_ms);
            assert!((actual.flakiness_rate - expected.flakiness_rate).abs() < 1e-12);
            assert_eq!(actual.last_run_ts, expected.last_run_ts);
        }
    }
}
