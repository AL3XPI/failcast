//! Git diff/history miner and Tree-sitter Python AST import graph collector.

pub mod ast_python;
pub mod git;

pub use ast_python::{
    CachedAstFile, CachedImportGraphResult, ImportGraph, ImportGraphBuilder,
    PythonImportGraphBuilder,
};
pub use git::{
    TempGitWorktree, extract_working_diff, hash_author, is_git_repo, mine_commit_history,
    mine_commit_history_from_ref,
};

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use failcast_core::DiffSummary;

    use super::*;

    fn git_cmd(dir: &Path, args: &[&str], ts: Option<i64>) {
        let mut cmd = Command::new("git");
        cmd.args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Failcast Tester")
            .env("GIT_AUTHOR_EMAIL", "tester@example.com")
            .env("GIT_COMMITTER_NAME", "Failcast Tester")
            .env("GIT_COMMITTER_EMAIL", "tester@example.com");
        if let Some(t) = ts {
            let date = format!("{t} +0000");
            cmd.env("GIT_AUTHOR_DATE", &date)
                .env("GIT_COMMITTER_DATE", &date);
        }
        let out = cmd.output().expect("run git command");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
        if args.first() == Some(&"init") {
            let cfg_out = Command::new("git")
                .args(["config", "core.autocrlf", "false"])
                .current_dir(dir)
                .output()
                .expect("configure git core.autocrlf");
            assert!(cfg_out.status.success(), "git config core.autocrlf failed");
        }
    }

    #[test]
    fn python_ast_resolves_src_layout_and_relative_hops() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        fs::create_dir_all(root.join("src/pkg")).expect("mkdir src/pkg");
        fs::create_dir_all(root.join("tests")).expect("mkdir tests");

        fs::write(root.join("src/pkg/__init__.py"), "").expect("write __init__.py");
        fs::write(
            root.join("src/pkg/b.py"),
            "def helper(x):\n    return x * 2\n",
        )
        .expect("write b.py");
        fs::write(
            root.join("src/pkg/a.py"),
            "from . import b\n\ndef compute(x):\n    return b.helper(x) + 1\n",
        )
        .expect("write a.py");
        fs::write(
            root.join("src/pkg/unrelated.py"),
            "def noop():\n    return None\n",
        )
        .expect("write unrelated.py");
        fs::write(
            root.join("tests/test_a.py"),
            "import pkg.a\n\ndef test_compute():\n    assert pkg.a.compute(2) == 5\n",
        )
        .expect("write test_a.py");

        let builder = PythonImportGraphBuilder::new();
        let graph = builder.build(root).expect("build import graph");

        assert_eq!(
            graph.shortest_hop("tests/test_a.py", "src/pkg/a.py"),
            Some(1),
            "test_a -> a must be 1 hop"
        );
        assert_eq!(
            graph.shortest_hop("tests/test_a.py", "src/pkg/b.py"),
            Some(2),
            "test_a -> b via relative import must be 2 hops"
        );
        assert_eq!(
            graph.shortest_hop("tests/test_a.py", "src/pkg/unrelated.py"),
            None,
            "test_a -> unrelated must be unreachable"
        );
        assert_eq!(
            graph.shortest_hop("tests/test_a.py", "tests/test_a.py"),
            Some(0),
            "test_a -> test_a must be 0 hops"
        );
        assert_eq!(
            graph.min_hop_to_any("tests/test_a.py", &["src/pkg/unrelated.py", "src/pkg/b.py"]),
            Some(2)
        );
    }

    #[test]
    fn git_miner_extracts_history_and_working_diff() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        git_cmd(root, &["init"], None);
        fs::create_dir_all(root.join("src")).expect("mkdir src");
        fs::create_dir_all(root.join("tests")).expect("mkdir tests");

        fs::write(
            root.join("src/calc.py"),
            "def add(a, b):\n    return a + b\n",
        )
        .expect("write calc.py");
        fs::write(
            root.join("tests/test_calc.py"),
            "def test_add():\n    assert True\n",
        )
        .expect("write test_calc.py");
        git_cmd(root, &["add", "."], None);
        git_cmd(root, &["commit", "-m", "initial"], Some(1_700_000_100));

        fs::write(
            root.join("src/calc.py"),
            "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
        )
        .expect("update calc.py");
        fs::write(
            root.join("tests/test_calc.py"),
            "def test_add():\n    assert True\n\ndef test_sub():\n    assert True\n",
        )
        .expect("update test_calc.py");
        git_cmd(root, &["add", "."], None);
        git_cmd(root, &["commit", "-m", "add sub"], Some(1_700_000_200));

        fs::write(
            root.join("src/calc.py"),
            "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n",
        )
        .expect("update calc.py again");
        git_cmd(root, &["add", "."], None);
        git_cmd(root, &["commit", "-m", "add mul"], Some(1_700_000_300));

        fs::write(
            root.join("src/calc.py"),
            "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n\ndef div(a, b):\n    return a / b\n",
        )
        .expect("working tree edit");

        let commits = mine_commit_history(root, 10).expect("mine commit history");
        assert_eq!(commits.len(), 3, "expected exactly 3 commits");

        let expected_author = hash_author("tester@example.com");
        assert_eq!(commits[0].commit_ts, 1_700_000_300);
        assert_eq!(commits[0].author_hash, expected_author);
        assert_eq!(commits[0].files_changed, 1);
        assert_eq!(commits[0].lines_added, 3);
        assert_eq!(commits[0].lines_deleted, 0);
        assert_eq!(commits[0].files[0].path, "src/calc.py");

        assert_eq!(commits[1].commit_ts, 1_700_000_200);
        assert_eq!(commits[1].files_changed, 2);
        assert_eq!(commits[1].parent_sha, Some(commits[2].commit_sha.clone()));

        assert_eq!(commits[2].commit_ts, 1_700_000_100);
        assert_eq!(commits[2].parent_sha, None);
        assert_eq!(commits[2].files_changed, 2);

        let calc_churn = commits
            .iter()
            .filter(|c| c.files.iter().any(|f| f.path == "src/calc.py"))
            .count();
        let test_churn = commits
            .iter()
            .filter(|c| c.files.iter().any(|f| f.path == "tests/test_calc.py"))
            .count();
        assert_eq!(calc_churn, 3);
        assert_eq!(test_churn, 2);

        let diff = extract_working_diff(root).expect("extract working diff");
        assert_eq!(diff.base_commit, Some(commits[0].commit_sha.clone()));
        assert_eq!(diff.files.len(), 1);
        assert_eq!(diff.files[0].path, "src/calc.py");
        assert_eq!(diff.files[0].lines_added, 3);
        assert_eq!(diff.files[0].lines_deleted, 0);
        assert_eq!(
            diff.diff_fingerprint,
            DiffSummary::compute_fingerprint(&diff.files)
        );
    }

    #[test]
    fn python_ast_incremental_cache_matches_cold_parse_and_handles_mutations() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        fs::create_dir_all(root.join("src/pkg")).expect("mkdir src/pkg");
        fs::create_dir_all(root.join("tests")).expect("mkdir tests");

        fs::write(root.join("src/pkg/__init__.py"), "").expect("write __init__.py");
        fs::write(
            root.join("src/pkg/b.py"),
            "def helper(x):\n    return x * 2\n",
        )
        .expect("write b.py");
        fs::write(
            root.join("src/pkg/a.py"),
            "from . import b, extra\n\ndef compute(x):\n    return b.helper(x) + 1\n",
        )
        .expect("write a.py");
        fs::write(
            root.join("tests/test_a.py"),
            "import pkg.a\n\ndef test_compute():\n    assert pkg.a.compute(2) == 5\n",
        )
        .expect("write test_a.py");

        let builder = PythonImportGraphBuilder::new();
        let mut cache: BTreeMap<String, CachedAstFile> = BTreeMap::new();

        // 1. Cold build vs first incremental build must match identically.
        let cold_1 = builder.build(root).expect("cold build 1");
        let inc_1 = builder
            .build_with_cache(root, &cache)
            .expect("incremental build 1");
        assert_eq!(inc_1.graph, cold_1);
        assert_eq!(inc_1.parsed_files, 4);
        assert_eq!(inc_1.cached_files, 0);
        assert!(inc_1.deleted_paths.is_empty());

        for entry in inc_1.upserts {
            cache.insert(entry.file_path.clone(), entry);
        }

        // 2. Second run parses 0 files (100% cache hit).
        let inc_2 = builder
            .build_with_cache(root, &cache)
            .expect("incremental build 2");
        assert_eq!(inc_2.graph, cold_1);
        assert_eq!(inc_2.parsed_files, 0);
        assert_eq!(inc_2.cached_files, 4);
        assert!(inc_2.upserts.is_empty());
        assert!(inc_2.deleted_paths.is_empty());

        // 3. Adding a sibling file referenced by unchanged a.py parses only the new file
        //    and still resolves the new edge accurately.
        fs::write(
            root.join("src/pkg/extra.py"),
            "def bonus():\n    return 10\n",
        )
        .expect("write extra.py");

        let cold_2 = builder.build(root).expect("cold build 2");
        let inc_3 = builder
            .build_with_cache(root, &cache)
            .expect("incremental build 3");
        assert_eq!(inc_3.graph, cold_2);
        assert_eq!(inc_3.parsed_files, 1);
        assert_eq!(inc_3.cached_files, 4);
        assert_eq!(inc_3.upserts.len(), 1);
        assert_eq!(inc_3.upserts[0].file_path, "src/pkg/extra.py");
        assert_eq!(
            inc_3
                .graph
                .shortest_hop("tests/test_a.py", "src/pkg/extra.py"),
            Some(2)
        );
        for entry in inc_3.upserts {
            cache.insert(entry.file_path.clone(), entry);
        }

        // 4. Modifying an existing file invalidates only that file.
        fs::write(
            root.join("src/pkg/extra.py"),
            "from . import leaf\n\ndef bonus():\n    return leaf.val()\n",
        )
        .expect("modify extra.py");
        fs::write(root.join("src/pkg/leaf.py"), "def val():\n    return 1\n")
            .expect("write leaf.py");

        let cold_3 = builder.build(root).expect("cold build 3");
        let inc_4 = builder
            .build_with_cache(root, &cache)
            .expect("incremental build 4");
        assert_eq!(inc_4.graph, cold_3);
        assert_eq!(inc_4.parsed_files, 2);
        assert_eq!(inc_4.cached_files, 4);
        assert_eq!(
            inc_4
                .graph
                .shortest_hop("tests/test_a.py", "src/pkg/leaf.py"),
            Some(3)
        );
        for entry in inc_4.upserts {
            cache.insert(entry.file_path.clone(), entry);
        }

        // 5. Deleting a file parses 0 files, reports deleted_paths, and updates graph hops.
        fs::remove_file(root.join("src/pkg/extra.py")).expect("remove extra.py");
        let cold_4 = builder.build(root).expect("cold build 4");
        let inc_5 = builder
            .build_with_cache(root, &cache)
            .expect("incremental build 5");
        assert_eq!(inc_5.graph, cold_4);
        assert_eq!(inc_5.parsed_files, 0);
        assert_eq!(inc_5.cached_files, 5);
        assert_eq!(inc_5.deleted_paths, vec!["src/pkg/extra.py".to_string()]);
        assert_eq!(
            inc_5
                .graph
                .shortest_hop("tests/test_a.py", "src/pkg/leaf.py"),
            None
        );
    }
}
