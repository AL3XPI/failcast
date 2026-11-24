//! Pure feature engineering, cold-start heuristic ranker, and cost-aware selector.
//!
//! Invariant 2: This crate is 100% pure — no disk I/O, no subprocesses, no DB,
//! and no `SystemTime::now()` calls.

pub mod features;
pub mod heuristic;
pub mod selector;

pub use features::{FEATURE_NAMES, NUM_FEATURES, extract_features};
pub use heuristic::{ast_hop_decay, score_heuristic, should_use_heuristic};
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
}
