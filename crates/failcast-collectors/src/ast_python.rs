use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::time::UNIX_EPOCH;

use failcast_core::{FailcastError, normalize_posix_path};
use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser};

const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".failcast",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "target",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
];

/// Directed file-level import graph using canonical repo-relative POSIX paths.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportGraph {
    edges: BTreeMap<String, BTreeSet<String>>,
}

impl ImportGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_edge(&mut self, from_file: impl Into<String>, to_file: impl Into<String>) {
        let from = normalize_posix_path(&from_file.into());
        let to = normalize_posix_path(&to_file.into());
        if from.is_empty() || to.is_empty() || from == to {
            return;
        }
        self.edges.entry(from).or_default().insert(to);
    }

    pub fn edges(&self) -> &BTreeMap<String, BTreeSet<String>> {
        &self.edges
    }

    pub fn imports_of(&self, file_path: &str) -> Option<&BTreeSet<String>> {
        let normalized = normalize_posix_path(file_path);
        self.edges.get(&normalized)
    }

    /// Computes shortest directed hop distance from `from_file` to `target_file`.
    /// Returns `Some(0)` if `from_file == target_file`, and `None` if unreachable.
    pub fn shortest_hop(&self, from_file: &str, target_file: &str) -> Option<usize> {
        let target = normalize_posix_path(target_file);
        self.min_hop_to_any(from_file, &[target])
    }

    /// Computes shortest directed hop distance via BFS from `from_file` to any file in
    /// `modified_files`. Returns `Some(0)` if `from_file` itself is modified, and `None`
    /// if no modified file is reachable.
    pub fn min_hop_to_any<S: AsRef<str>>(
        &self,
        from_file: &str,
        modified_files: &[S],
    ) -> Option<usize> {
        if modified_files.is_empty() {
            return None;
        }

        let start = normalize_posix_path(from_file);
        let targets: HashSet<String> = modified_files
            .iter()
            .map(|s| normalize_posix_path(s.as_ref()))
            .collect();

        if targets.contains(&start) {
            return Some(0);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        visited.insert(start.clone());
        queue.push_back((start, 0));

        while let Some((node, dist)) = queue.pop_front() {
            if let Some(neighbors) = self.edges.get(&node) {
                for next in neighbors {
                    if targets.contains(next) {
                        return Some(dist + 1);
                    }
                    if visited.insert(next.clone()) {
                        queue.push_back((next.clone(), dist + 1));
                    }
                }
            }
        }

        None
    }
}

/// Language-agnostic builder trait for file-level import dependency graphs.
pub trait ImportGraphBuilder: Send + Sync {
    fn build(&self, repo_root: &Path) -> Result<ImportGraph, FailcastError>;
}

/// Cached raw Python import payload for a single file, keyed by `(file_size, mtime_ns)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedAstFile {
    pub file_path: String,
    pub file_size: u64,
    pub mtime_ns: i64,
    pub imports_json: String,
}

/// Output of an incremental AST import graph build against a persisted file cache.
#[derive(Debug, Clone)]
pub struct CachedImportGraphResult {
    pub graph: ImportGraph,
    pub upserts: Vec<CachedAstFile>,
    pub deleted_paths: Vec<String>,
    pub parsed_files: usize,
    pub cached_files: usize,
}

/// Tree-sitter backed Python import graph builder supporting `import a.b`,
/// `from a.b import c`, relative imports (`.`, `..`), `__init__.py` packages,
/// and both root and `src/` repository layouts.
#[derive(Debug, Clone, Default)]
pub struct PythonImportGraphBuilder;

impl PythonImportGraphBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Builds an `ImportGraph` incrementally using `existing_cache`.
    ///
    /// Unchanged files (matching both `file_size` and `mtime_ns`) reuse their cached raw
    /// `ParsedImport` list without invoking `tree-sitter`, while edges are resolved against the
    /// live repository file set so adding or deleting sibling `.py` files updates hops accurately.
    pub fn build_with_cache(
        &self,
        repo_root: &Path,
        existing_cache: &BTreeMap<String, CachedAstFile>,
    ) -> Result<CachedImportGraphResult, FailcastError> {
        let mut files_with_meta = BTreeMap::new();
        collect_python_files_with_meta(repo_root, repo_root, &mut files_with_meta)?;

        let py_files: BTreeSet<String> = files_with_meta.keys().cloned().collect();
        let module_index = build_module_index(&py_files);

        let mut parser: Option<Parser> = None;
        let mut graph = ImportGraph::new();
        let mut upserts = Vec::new();
        let mut parsed_files = 0usize;
        let mut cached_files = 0usize;

        for (rel_path, &(file_size, mtime_ns)) in &files_with_meta {
            let cached_imports = existing_cache
                .get(rel_path)
                .filter(|entry| entry.file_size == file_size && entry.mtime_ns == mtime_ns)
                .and_then(|entry| {
                    serde_json::from_str::<Vec<ParsedImport>>(&entry.imports_json).ok()
                });

            let imports = if let Some(imports) = cached_imports {
                cached_files += 1;
                imports
            } else {
                let abs_path = repo_root.join(rel_path);
                let source = match fs::read_to_string(&abs_path) {
                    Ok(text) => text,
                    Err(_) => continue,
                };

                let ts_parser = match parser.as_mut() {
                    Some(p) => p,
                    None => {
                        let mut p = Parser::new();
                        p.set_language(&tree_sitter_python::LANGUAGE.into())
                            .map_err(|e| {
                                FailcastError::Parse(format!(
                                    "failed to load tree-sitter-python: {e}"
                                ))
                            })?;
                        parser.insert(p)
                    }
                };

                let Some(tree) = ts_parser.parse(&source, None) else {
                    continue;
                };

                parsed_files += 1;
                let imports = extract_imports_from_tree(tree.root_node(), source.as_bytes());
                let imports_json = serde_json::to_string(&imports).map_err(|e| {
                    FailcastError::Parse(format!("failed to serialize AST imports: {e}"))
                })?;
                upserts.push(CachedAstFile {
                    file_path: rel_path.clone(),
                    file_size,
                    mtime_ns,
                    imports_json,
                });
                imports
            };

            for stmt in imports {
                for target in resolve_import(rel_path, &stmt, &module_index, &py_files) {
                    graph.add_edge(rel_path.clone(), target);
                }
            }
        }

        let deleted_paths = existing_cache
            .keys()
            .filter(|path| !files_with_meta.contains_key(*path))
            .cloned()
            .collect();

        Ok(CachedImportGraphResult {
            graph,
            upserts,
            deleted_paths,
            parsed_files,
            cached_files,
        })
    }
}

impl ImportGraphBuilder for PythonImportGraphBuilder {
    fn build(&self, repo_root: &Path) -> Result<ImportGraph, FailcastError> {
        let mut py_files = BTreeSet::new();
        collect_python_files(repo_root, repo_root, &mut py_files)?;

        let module_index = build_module_index(&py_files);
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .map_err(|e| FailcastError::Parse(format!("failed to load tree-sitter-python: {e}")))?;

        let mut graph = ImportGraph::new();
        for rel_path in &py_files {
            let abs_path = repo_root.join(rel_path);
            let source = match fs::read_to_string(&abs_path) {
                Ok(text) => text,
                Err(_) => continue,
            };

            let Some(tree) = parser.parse(&source, None) else {
                continue;
            };

            let imports = extract_imports_from_tree(tree.root_node(), source.as_bytes());
            for stmt in imports {
                for target in resolve_import(rel_path, &stmt, &module_index, &py_files) {
                    graph.add_edge(rel_path.clone(), target);
                }
            }
        }

        Ok(graph)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ParsedImport {
    /// `import a.b` or `import a.b as c`
    Module(String),
    /// `from <dots><module> import <names>`
    From {
        relative_level: usize,
        module: Option<String>,
        names: Vec<String>,
    },
}

fn collect_python_files(
    repo_root: &Path,
    dir: &Path,
    out: &mut BTreeSet<String>,
) -> Result<(), FailcastError> {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if dir == repo_root => return Err(e.into()),
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if file_type.is_dir() {
            if IGNORED_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            collect_python_files(repo_root, &entry.path(), out)?;
        } else if file_type.is_file() && name_str.ends_with(".py") {
            let rel = entry
                .path()
                .strip_prefix(repo_root)
                .map_err(|e| FailcastError::Parse(format!("failed to strip repo root: {e}")))?
                .to_string_lossy()
                .to_string();
            out.insert(normalize_posix_path(&rel));
        }
    }

    Ok(())
}

fn collect_python_files_with_meta(
    repo_root: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, (u64, i64)>,
) -> Result<(), FailcastError> {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if dir == repo_root => return Err(e.into()),
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if file_type.is_dir() {
            if IGNORED_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            collect_python_files_with_meta(repo_root, &entry.path(), out)?;
        } else if file_type.is_file() && name_str.ends_with(".py") {
            let rel = entry
                .path()
                .strip_prefix(repo_root)
                .map_err(|e| FailcastError::Parse(format!("failed to strip repo root: {e}")))?
                .to_string_lossy()
                .to_string();
            let meta = entry.metadata()?;
            let file_size = meta.len();
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos().min(i64::MAX as u128) as i64)
                .unwrap_or(0);
            out.insert(normalize_posix_path(&rel), (file_size, mtime_ns));
        }
    }

    Ok(())
}

/// Builds a map from dotted Python module name to one or more matching repo-relative files.
/// Registers both repo-root module paths and `src/`-stripped module paths.
fn build_module_index(py_files: &BTreeSet<String>) -> BTreeMap<String, Vec<String>> {
    let mut index: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for path in py_files {
        if let Some(module) = path_to_module(path) {
            push_unique(index.entry(module).or_default(), path);
        }
        if let Some(stripped) = path.strip_prefix("src/")
            && let Some(module) = path_to_module(stripped)
        {
            push_unique(index.entry(module).or_default(), path);
        }
    }

    index
}

fn push_unique(vec: &mut Vec<String>, value: &str) {
    if !vec.iter().any(|v| v == value) {
        vec.push(value.to_string());
    }
}

fn path_to_module(rel_path: &str) -> Option<String> {
    let without_ext = rel_path.strip_suffix(".py")?;
    let mut parts: Vec<&str> = without_ext
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    if parts.last().copied() == Some("__init__") {
        parts.pop();
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("."))
}

fn extract_imports_from_tree(root: Node<'_>, source: &[u8]) -> Vec<ParsedImport> {
    let mut imports = Vec::new();
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        match node.kind() {
            "import_statement" => {
                extract_import_statement(node, source, &mut imports);
            }
            "import_from_statement" => {
                if let Some(parsed) = extract_import_from_statement(node, source) {
                    imports.push(parsed);
                }
            }
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    stack.push(child);
                }
            }
        }
    }

    imports
}

fn extract_import_statement(node: Node<'_>, source: &[u8], out: &mut Vec<ParsedImport>) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "dotted_name" => {
                if let Some(text) = node_text(child, source) {
                    out.push(ParsedImport::Module(text));
                }
            }
            "aliased_import" => {
                if let Some(name_node) = child.child_by_field_name("name")
                    && let Some(text) = node_text(name_node, source)
                {
                    out.push(ParsedImport::Module(text));
                }
            }
            _ => {}
        }
    }
}

fn extract_import_from_statement(node: Node<'_>, source: &[u8]) -> Option<ParsedImport> {
    let module_node = node.child_by_field_name("module_name")?;
    let (relative_level, module) = match module_node.kind() {
        "dotted_name" => (0, node_text(module_node, source)),
        "relative_import" => {
            let mut dots = 0usize;
            let mut submod = None;
            let mut cursor = module_node.walk();
            for child in module_node.named_children(&mut cursor) {
                match child.kind() {
                    "import_prefix" => {
                        if let Some(prefix) = node_text(child, source) {
                            dots = prefix.chars().filter(|&c| c == '.').count();
                        }
                    }
                    "dotted_name" => {
                        submod = node_text(child, source);
                    }
                    _ => {}
                }
            }
            (dots, submod)
        }
        _ => return None,
    };

    let mut names = Vec::new();
    let mut cursor = node.walk();
    for child in node.children_by_field_name("name", &mut cursor) {
        match child.kind() {
            "dotted_name" | "identifier" => {
                if let Some(text) = node_text(child, source) {
                    names.push(text);
                }
            }
            "aliased_import" => {
                if let Some(name_node) = child.child_by_field_name("name")
                    && let Some(text) = node_text(name_node, source)
                {
                    names.push(text);
                }
            }
            _ => {}
        }
    }

    Some(ParsedImport::From {
        relative_level,
        module,
        names,
    })
}

fn node_text(node: Node<'_>, source: &[u8]) -> Option<String> {
    node.utf8_text(source)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn resolve_import(
    current_file: &str,
    stmt: &ParsedImport,
    module_index: &BTreeMap<String, Vec<String>>,
    py_files: &BTreeSet<String>,
) -> Vec<String> {
    let mut resolved = Vec::new();

    match stmt {
        ParsedImport::Module(mod_name) => {
            if let Some(targets) = module_index.get(mod_name) {
                for t in targets {
                    push_unique(&mut resolved, t);
                }
            }
        }
        ParsedImport::From {
            relative_level,
            module,
            names,
        } => {
            if *relative_level == 0 {
                if let Some(base_mod) = module {
                    for sym in names {
                        let candidate = format!("{base_mod}.{sym}");
                        if let Some(targets) = module_index.get(&candidate) {
                            for t in targets {
                                push_unique(&mut resolved, t);
                            }
                        }
                    }
                    if let Some(targets) = module_index.get(base_mod) {
                        for t in targets {
                            push_unique(&mut resolved, t);
                        }
                    }
                }
            } else {
                // PEP 328 relative import resolution from current file's package directory.
                let mut dir_parts: Vec<&str> = current_file.split('/').collect();
                dir_parts.pop();

                let ascents = relative_level.saturating_sub(1);
                if ascents > dir_parts.len() {
                    return resolved;
                }
                dir_parts.truncate(dir_parts.len() - ascents);

                let mut base_parts: Vec<String> = dir_parts.into_iter().map(String::from).collect();
                if let Some(submod) = module {
                    for seg in submod.split('.').filter(|s| !s.is_empty()) {
                        base_parts.push(seg.to_string());
                    }
                }

                let mut matched_submodule = false;
                for sym in names {
                    let mut sym_parts = base_parts.clone();
                    for seg in sym.split('.').filter(|s| !s.is_empty()) {
                        sym_parts.push(seg.to_string());
                    }
                    if let Some(target) = resolve_parts_as_file(&sym_parts, py_files) {
                        push_unique(&mut resolved, &target);
                        matched_submodule = true;
                    }
                }

                if (module.is_some() || !matched_submodule)
                    && let Some(target) = resolve_parts_as_file(&base_parts, py_files)
                {
                    push_unique(&mut resolved, &target);
                }
            }
        }
    }

    resolved
}

fn resolve_parts_as_file(parts: &[String], py_files: &BTreeSet<String>) -> Option<String> {
    if parts.is_empty() {
        return None;
    }
    let joined = parts.join("/");
    let module_file = format!("{joined}.py");
    if py_files.contains(&module_file) {
        return Some(module_file);
    }
    let init_file = format!("{joined}/__init__.py");
    if py_files.contains(&init_file) {
        return Some(init_file);
    }
    None
}
