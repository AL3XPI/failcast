use std::collections::{BTreeMap, BTreeSet, VecDeque};

use pts_core::{
    CoFailCounts, CommitRecord, DiffSummary, HistoricalRun, HistorySnapshot, TestCase, TestHistory,
    TestStatus,
};

use crate::features::{NUM_FEATURES, extract_features};

const SECONDS_PER_30_DAYS: i64 = 30 * 86_400;
const MAX_COCHANGE_FILES_PER_COMMIT: usize = 200;
const EWMA_ALPHA_10: f64 = 0.2;
const EWMA_ALPHA_50: f64 = 0.04;

/// A single non-skip historical `(run, test)` example with leakage-free features
/// extracted strictly from runs prior to `started_at`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRow {
    pub run_id: String,
    pub started_at: i64,
    pub test_id: String,
    pub features: [f64; NUM_FEATURES],
    pub label: f64,
    pub sampling_propensity: f64,
}

#[derive(Debug, Clone, Default)]
struct TestAcc {
    total_runs: u64,
    total_failures: u64,
    ewma_10: f64,
    ewma_50: f64,
    runs_since_last_fail: Option<u64>,
    durations: Vec<u64>,
    last_run_ts: Option<i64>,
    fp_flags: BTreeMap<String, (bool, bool)>,
}

/// Pure in-memory chronological history accumulator.
///
/// Streams historical runs and git commits strictly ascending by timestamp so that
/// features extracted for any run at `started_at` use ONLY state strictly prior to
/// `started_at`, matching `pts_store::Store::load_history_snapshot` at identical cutoffs.
#[derive(Debug, Clone)]
pub struct HistoryAccumulator {
    total_runs: usize,
    total_failures: usize,
    repo_durations: Vec<u64>,
    tests: BTreeMap<String, TestAcc>,
    cofail_counts: BTreeMap<(String, String), CoFailCounts>,

    commits: Vec<CommitRecord>,
    next_commit_idx: usize,
    file_commit_counts: BTreeMap<String, u32>,
    pair_commit_counts: BTreeMap<(String, String), u32>,
    churn_window: VecDeque<(i64, Vec<String>)>,
    file_churn_30d: BTreeMap<String, u32>,
}

impl HistoryAccumulator {
    /// Creates a new empty accumulator over a set of mined git commits.
    pub fn new(mut commits: Vec<CommitRecord>) -> Self {
        commits.sort_by(|a, b| {
            a.commit_ts
                .cmp(&b.commit_ts)
                .then_with(|| a.commit_sha.cmp(&b.commit_sha))
        });

        Self {
            total_runs: 0,
            total_failures: 0,
            repo_durations: Vec::new(),
            tests: BTreeMap::new(),
            cofail_counts: BTreeMap::new(),
            commits,
            next_commit_idx: 0,
            file_commit_counts: BTreeMap::new(),
            pair_commit_counts: BTreeMap::new(),
            churn_window: VecDeque::new(),
            file_churn_30d: BTreeMap::new(),
        }
    }

    /// Advances git commit statistics to include all commits with `commit_ts < as_of_ts`
    /// and evicts commits older than `[as_of_ts - 30d, as_of_ts)` from 30-day churn.
    pub fn advance_commits(&mut self, as_of_ts: i64) {
        let cutoff_30d = as_of_ts.saturating_sub(SECONDS_PER_30_DAYS);

        while self.next_commit_idx < self.commits.len()
            && self.commits[self.next_commit_idx].commit_ts < as_of_ts
        {
            let commit = &self.commits[self.next_commit_idx];
            let unique_files: Vec<String> = commit
                .files
                .iter()
                .map(|f| f.path.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();

            for path in &unique_files {
                *self.file_commit_counts.entry(path.clone()).or_insert(0) += 1;
            }
            if unique_files.len() <= MAX_COCHANGE_FILES_PER_COMMIT {
                for i in 0..unique_files.len() {
                    for j in (i + 1)..unique_files.len() {
                        *self
                            .pair_commit_counts
                            .entry((unique_files[i].clone(), unique_files[j].clone()))
                            .or_insert(0) += 1;
                    }
                }
            }
            if commit.commit_ts >= cutoff_30d && !unique_files.is_empty() {
                for path in &unique_files {
                    *self.file_churn_30d.entry(path.clone()).or_insert(0) += 1;
                }
                self.churn_window
                    .push_back((commit.commit_ts, unique_files));
            }

            self.next_commit_idx += 1;
        }

        while let Some((ts, _)) = self.churn_window.front() {
            if *ts < cutoff_30d {
                let (_, evicted_files) = self.churn_window.pop_front().expect("checked front");
                for path in evicted_files {
                    if let Some(count) = self.file_churn_30d.get_mut(&path) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            self.file_churn_30d.remove(&path);
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Incorporates all outcomes and diff files from `run` into accumulated state.
    pub fn record_run(&mut self, run: &HistoricalRun) {
        self.total_runs += 1;

        let mut sorted_outcomes: Vec<_> = run.outcomes.iter().collect();
        sorted_outcomes.sort_by(|a, b| a.test_id.cmp(&b.test_id));

        let unique_diff_files: Vec<&str> = run
            .diff_files
            .iter()
            .map(|f| f.path.as_str())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        for outcome in sorted_outcomes {
            let entry = self.tests.entry(outcome.test_id.clone()).or_default();
            entry.total_runs += 1;
            entry.last_run_ts = Some(match entry.last_run_ts {
                Some(prev) => prev.max(run.started_at),
                None => run.started_at,
            });

            match outcome.status {
                TestStatus::Fail | TestStatus::Error => {
                    entry.total_failures += 1;
                    self.total_failures += 1;
                    entry.ewma_10 = EWMA_ALPHA_10 + (1.0 - EWMA_ALPHA_10) * entry.ewma_10;
                    entry.ewma_50 = EWMA_ALPHA_50 + (1.0 - EWMA_ALPHA_50) * entry.ewma_50;
                    entry.runs_since_last_fail = Some(0);
                    entry.durations.push(outcome.duration_ms);
                    self.repo_durations.push(outcome.duration_ms);
                    entry
                        .fp_flags
                        .entry(run.diff_fingerprint.clone())
                        .or_insert((false, false))
                        .1 = true;

                    for &file_path in &unique_diff_files {
                        let co = self
                            .cofail_counts
                            .entry((outcome.test_id.clone(), file_path.to_string()))
                            .or_default();
                        co.co_runs += 1;
                        co.co_fails += 1;
                    }
                }
                TestStatus::Pass => {
                    entry.ewma_10 *= 1.0 - EWMA_ALPHA_10;
                    entry.ewma_50 *= 1.0 - EWMA_ALPHA_50;
                    if let Some(since) = entry.runs_since_last_fail {
                        entry.runs_since_last_fail = Some(since + 1);
                    }
                    entry.durations.push(outcome.duration_ms);
                    self.repo_durations.push(outcome.duration_ms);
                    entry
                        .fp_flags
                        .entry(run.diff_fingerprint.clone())
                        .or_insert((false, false))
                        .0 = true;

                    for &file_path in &unique_diff_files {
                        let co = self
                            .cofail_counts
                            .entry((outcome.test_id.clone(), file_path.to_string()))
                            .or_default();
                        co.co_runs += 1;
                    }
                }
                TestStatus::Skip => {
                    // Matches Store::load_history_snapshot: skips update total_runs and
                    // last_run_ts but do not bias failure rates, durations, flakiness,
                    // or co-failure statistics.
                }
            }
        }
    }

    /// Produces a point-in-time `HistorySnapshot` matching `Store::load_history_snapshot(as_of_ts)`.
    /// Note: callers must ensure only runs with `started_at < as_of_ts` have been recorded.
    pub fn snapshot(&mut self, as_of_ts: i64) -> HistorySnapshot {
        self.advance_commits(as_of_ts);

        let median_repo_test_ms = median_u64(&mut self.repo_durations);
        let mut test_history = BTreeMap::new();

        for (test_id, acc) in &mut self.tests {
            let total_fps = acc.fp_flags.len();
            let flaky_fps = acc
                .fp_flags
                .values()
                .filter(|&&(has_pass, has_fail)| has_pass && has_fail)
                .count();
            let flakiness_rate = if total_fps == 0 {
                0.0
            } else {
                flaky_fps as f64 / total_fps as f64
            };
            let median_duration_ms = median_u64(&mut acc.durations);

            test_history.insert(
                test_id.clone(),
                TestHistory {
                    test_id: test_id.clone(),
                    total_runs: acc.total_runs,
                    total_failures: acc.total_failures,
                    ewma_fail_10: acc.ewma_10,
                    ewma_fail_50: acc.ewma_50,
                    runs_since_last_fail: acc.runs_since_last_fail,
                    median_duration_ms,
                    flakiness_rate,
                    last_run_ts: acc.last_run_ts,
                },
            );
        }

        HistorySnapshot {
            as_of_ts,
            total_runs: self.total_runs,
            total_failures: self.total_failures,
            median_repo_test_ms,
            test_history,
            cofail_counts: self.cofail_counts.clone(),
            file_commit_counts: self.file_commit_counts.clone(),
            pair_commit_counts: self.pair_commit_counts.clone(),
            file_churn_30d: self.file_churn_30d.clone(),
        }
    }

    /// Extracts labeled feature rows for all non-skip outcomes in `run` using the
    /// provided point-in-time `snapshot` (strictly prior to `run.started_at`).
    pub fn extract_rows_from_snapshot(
        run: &HistoricalRun,
        snapshot: &HistorySnapshot,
    ) -> Vec<TrainingRow> {
        let diff = DiffSummary {
            base_commit: run.commit_sha.clone(),
            diff_fingerprint: run.diff_fingerprint.clone(),
            files: run.diff_files.clone(),
        };

        let mut sorted_outcomes: Vec<_> = run.outcomes.iter().collect();
        sorted_outcomes.sort_by(|a, b| a.test_id.cmp(&b.test_id));

        let mut rows = Vec::new();
        for outcome in sorted_outcomes {
            let label = match outcome.status {
                TestStatus::Pass => 0.0,
                TestStatus::Fail | TestStatus::Error => 1.0,
                TestStatus::Skip => continue,
            };

            let tc = TestCase {
                test_id: outcome.test_id.clone(),
                file_path: outcome.file_path.clone(),
            };
            let features = extract_features(&tc, &diff, snapshot, outcome.ast_min_hop);
            rows.push(TrainingRow {
                run_id: run.run_id.clone(),
                started_at: run.started_at,
                test_id: outcome.test_id.clone(),
                features,
                label,
                sampling_propensity: outcome.sampling_propensity,
            });
        }
        rows
    }

    /// Streams historical runs strictly ascending by `(started_at, run_id)` and builds
    /// leakage-free feature rows for every non-skip outcome.
    ///
    /// Runs sharing an identical `started_at` timestamp are extracted against the same
    /// strictly-prior snapshot before any of them updates the accumulator.
    pub fn build_dataset(runs: &[HistoricalRun], commits: &[CommitRecord]) -> Vec<TrainingRow> {
        let mut sorted: Vec<&HistoricalRun> = runs.iter().collect();
        sorted.sort_by(|a, b| {
            a.started_at
                .cmp(&b.started_at)
                .then_with(|| a.run_id.cmp(&b.run_id))
        });

        let mut acc = Self::new(commits.to_vec());
        let mut dataset = Vec::new();

        let mut i = 0;
        while i < sorted.len() {
            let ts = sorted[i].started_at;
            let mut j = i + 1;
            while j < sorted.len() && sorted[j].started_at == ts {
                j += 1;
            }

            let snapshot = acc.snapshot(ts);
            for &run in &sorted[i..j] {
                dataset.extend(Self::extract_rows_from_snapshot(run, &snapshot));
            }
            for &run in &sorted[i..j] {
                acc.record_run(run);
            }

            i = j;
        }

        dataset
    }
}

fn median_u64(values: &mut [u64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid] as f64)
    } else {
        Some((values[mid - 1] as f64 + values[mid] as f64) / 2.0)
    }
}
