use std::collections::BTreeSet;

use pts_core::{DiffSummary, HistorySnapshot, TestCase};

pub const NUM_FEATURES: usize = 16;

pub const IDX_AST_MIN_HOP: usize = 0;
pub const IDX_DIRECT_TEST_MODIFIED: usize = 1;
pub const IDX_PATH_JACCARD_MAX: usize = 2;
pub const IDX_SAME_DIR_OVERLAP: usize = 3;
pub const IDX_COFAIL_RATE_MAX: usize = 4;
pub const IDX_COCHANGE_FREQ_MAX: usize = 5;
pub const IDX_EWMA_FAIL_10: usize = 6;
pub const IDX_EWMA_FAIL_50: usize = 7;
pub const IDX_RUNS_SINCE_LAST_FAIL: usize = 8;
pub const IDX_TOTAL_TEST_RUNS: usize = 9;
pub const IDX_MEDIAN_DURATION_MS: usize = 10;
pub const IDX_FLAKINESS_RATE: usize = 11;
pub const IDX_DIFF_FILES_COUNT: usize = 12;
pub const IDX_DIFF_LINES_TOTAL: usize = 13;
pub const IDX_DIFF_TOP_DIRS_COUNT: usize = 14;
pub const IDX_TOUCHED_FILE_CHURN_30D: usize = 15;

pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "ast_min_hop",
    "direct_test_modified",
    "path_jaccard_max",
    "same_dir_overlap",
    "cofail_rate_max",
    "cochange_freq_max",
    "ewma_fail_10",
    "ewma_fail_50",
    "runs_since_last_fail",
    "total_test_runs",
    "median_duration_ms",
    "flakiness_rate",
    "diff_files_count",
    "diff_lines_total",
    "diff_top_dirs_count",
    "touched_file_churn_30d",
];

/// Pure feature extraction implementing all 16 features from SPEC.md Section 5.1
/// in exact index order. Undefined or missing values are encoded as `f64::NAN`.
pub fn extract_features(
    test: &TestCase,
    diff: &DiffSummary,
    history: &HistorySnapshot,
    ast_min_hop: Option<usize>,
) -> [f64; NUM_FEATURES] {
    let f_ast_min_hop = ast_min_hop.map(|h| h as f64).unwrap_or(f64::NAN);

    let f_direct_test_modified = if diff.files.iter().any(|f| f.path == test.file_path) {
        1.0
    } else {
        0.0
    };

    let test_tokens = path_tokens(&test.file_path);
    let test_parent = parent_dir(&test.file_path);

    let mut path_jaccard_max = 0.0_f64;
    let mut same_dir_overlap = 0.0_f64;
    let mut cofail_rate_max = 0.0_f64;
    let mut cochange_freq_max = 0.0_f64;
    let mut diff_lines_sum: u64 = 0;
    let mut top_dirs: BTreeSet<&str> = BTreeSet::new();
    let mut max_churn_30d: u32 = 0;

    for file in &diff.files {
        let mod_tokens = path_tokens(&file.path);
        let jacc = jaccard_similarity(&test_tokens, &mod_tokens);
        if jacc > path_jaccard_max {
            path_jaccard_max = jacc;
        }

        if parent_dir(&file.path) == test_parent {
            same_dir_overlap = 1.0;
        }

        let cofail = history.cofail_rate(&test.test_id, &file.path);
        if cofail > cofail_rate_max {
            cofail_rate_max = cofail;
        }

        let cochange = history.cochange_freq(&test.file_path, &file.path);
        if cochange > cochange_freq_max {
            cochange_freq_max = cochange;
        }

        diff_lines_sum = diff_lines_sum
            .saturating_add(u64::from(file.lines_added))
            .saturating_add(u64::from(file.lines_deleted));
        top_dirs.insert(top_dir(&file.path));

        let churn = history.churn_30d(&file.path);
        if churn > max_churn_30d {
            max_churn_30d = churn;
        }
    }

    let test_hist = history.test_history.get(&test.test_id);
    let ewma_fail_10 = test_hist.map(|h| h.ewma_fail_10).unwrap_or(0.0);
    let ewma_fail_50 = test_hist.map(|h| h.ewma_fail_50).unwrap_or(0.0);
    let runs_since_last_fail = test_hist
        .and_then(|h| h.runs_since_last_fail)
        .map(|n| n as f64)
        .unwrap_or(f64::NAN);
    let total_test_runs = test_hist.map(|h| h.total_runs as f64).unwrap_or(0.0);
    let median_duration_ms = test_hist
        .and_then(|h| h.median_duration_ms)
        .unwrap_or(f64::NAN);
    let flakiness_rate = test_hist.map(|h| h.flakiness_rate).unwrap_or(0.0);

    let diff_files_count = diff.files.len() as f64;
    let diff_lines_total = (diff_lines_sum as f64).ln_1p();
    let diff_top_dirs_count = top_dirs.len() as f64;
    let touched_file_churn_30d = f64::from(max_churn_30d);

    [
        f_ast_min_hop,
        f_direct_test_modified,
        path_jaccard_max,
        same_dir_overlap,
        cofail_rate_max,
        cochange_freq_max,
        ewma_fail_10,
        ewma_fail_50,
        runs_since_last_fail,
        total_test_runs,
        median_duration_ms,
        flakiness_rate,
        diff_files_count,
        diff_lines_total,
        diff_top_dirs_count,
        touched_file_churn_30d,
    ]
}

/// Tokenizes a POSIX repo-relative file path into lowercase stem/path tokens,
/// stripping the file extension and dropping boilerplate layout tokens (`src`, `test`, `tests`)
/// unless doing so would leave the token set empty.
pub fn path_tokens(path: &str) -> BTreeSet<String> {
    let without_ext = match path.rsplit_once('.') {
        Some((prefix, ext)) if !ext.contains('/') && !prefix.is_empty() => prefix,
        _ => path,
    };

    let raw_tokens: BTreeSet<String> = without_ext
        .split(['/', '_', '-', '.'])
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    let filtered: BTreeSet<String> = raw_tokens
        .iter()
        .filter(|t| !matches!(t.as_str(), "src" | "test" | "tests"))
        .cloned()
        .collect();

    if filtered.is_empty() {
        raw_tokens
    } else {
        filtered
    }
}

pub fn jaccard_similarity(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

pub fn parent_dir(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((parent, _)) => parent,
        None => "",
    }
}

pub fn top_dir(path: &str) -> &str {
    match path.split_once('/') {
        Some((first, _)) if !first.is_empty() => first,
        _ => ".",
    }
}
