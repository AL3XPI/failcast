//! Local MCP server over stdio using `rmcp`.
//!
//! Invariant 5 (Stdio MCP Purity): `stdout` is owned exclusively by JSON-RPC framing.
//! All tracing logs, warnings, and runner/git subprocess outputs MUST go to `stderr` or
//! be captured via pipes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use pts_core::{Config, SelectionReason};
use rmcp::{
    Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::service::{
    ExplainReport, FailureSummary, SelectionRequest, explain_test, run_selected_tests,
    select_for_working_tree,
};

/// JSON Schema-compatible selection reason matching `pts_core::SelectionReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpSelectionReason {
    Full,
    Heuristic,
    Ranked,
    Explore,
}

impl From<SelectionReason> for McpSelectionReason {
    fn from(reason: SelectionReason) -> Self {
        match reason {
            SelectionReason::Full => Self::Full,
            SelectionReason::Heuristic => Self::Heuristic,
            SelectionReason::Ranked => Self::Ranked,
            SelectionReason::Explore => Self::Explore,
        }
    }
}

/// Input parameters for the `select_tests` MCP tool (SPEC.md Section 6.2).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SelectTestsParams {
    /// Time budget in seconds (defaults to repo config `default_budget_secs`, 15s).
    #[serde(default)]
    pub budget_seconds: Option<u64>,
    /// Maximum number of tests to select (mutually exclusive with `target_recall`).
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Target failure recall in `(0, 1)` (mutually exclusive with `top_k`).
    #[serde(default)]
    pub target_recall: Option<f64>,
}

/// Selected test entry returned by `select_tests`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct McpSelectedTest {
    pub test_id: String,
    pub predicted_fail_prob: f64,
    pub est_duration_ms: u64,
    pub reason: McpSelectionReason,
}

/// Structured output for the `select_tests` MCP tool (SPEC.md Section 6.2).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SelectTestsOutput {
    pub model_mode: String,
    pub diff_files_count: usize,
    pub total_discovered_tests: usize,
    pub selected_count: usize,
    pub estimated_duration_ms: u64,
    pub selected: Vec<McpSelectedTest>,
}

/// Input parameters for the `run_fast_tests` MCP tool (SPEC.md Section 6.2).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct RunFastTestsParams {
    /// Time budget in seconds (defaults to repo config `default_budget_secs`, 15s).
    #[serde(default)]
    pub budget_seconds: Option<u64>,
    /// Maximum number of tests to select.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// If true, run the entire discovered test suite instead of selecting a subset.
    #[serde(default)]
    pub run_all: bool,
}

/// Structured output for the `run_fast_tests` MCP tool (SPEC.md Section 6.2).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunFastTestsOutput {
    pub run_id: String,
    pub model_mode: String,
    pub passed: usize,
    pub failed: usize,
    pub errored: usize,
    pub skipped: usize,
    pub total_duration_ms: u64,
    pub failures: Vec<FailureSummary>,
}

/// Input parameters for the `explain_selection` MCP tool (SPEC.md Section 6.2).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExplainSelectionParams {
    /// Canonical test identifier (e.g. `"tests/test_auth.py::test_refresh_token"`).
    pub test_id: String,
}

/// Local stdio MCP server exposing PTS selection, execution, and explainability tools.
#[derive(Debug, Clone)]
pub struct PtsMcpServer {
    repo_root: Arc<PathBuf>,
    tool_router: ToolRouter<Self>,
}

impl PtsMcpServer {
    pub fn new(repo_root: PathBuf) -> Self {
        Self {
            repo_root: Arc::new(repo_root),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router(router = tool_router)]
impl PtsMcpServer {
    /// Dry-run test selection for the current working-tree git diff (does not execute tests).
    #[tool(
        name = "select_tests",
        description = "Dry-run predictive test selection for the current working-tree git diff without executing tests."
    )]
    pub async fn select_tests(
        &self,
        Parameters(params): Parameters<SelectTestsParams>,
    ) -> Result<Json<SelectTestsOutput>, String> {
        if params.top_k.is_some() && params.target_recall.is_some() {
            return Err("specify at most one of `top_k` or `target_recall`".to_string());
        }

        let repo_root = Arc::clone(&self.repo_root);
        let output = tokio::task::spawn_blocking(move || -> Result<SelectTestsOutput> {
            let request = if let Some(top_k) = params.top_k {
                SelectionRequest::TopK { top_k }
            } else if let Some(target_recall) = params.target_recall {
                SelectionRequest::TargetRecall {
                    target_recall,
                    max_budget_seconds: params.budget_seconds,
                }
            } else {
                let budget_seconds = match params.budget_seconds {
                    Some(secs) => secs,
                    None => Config::load(&repo_root)?.default_budget_secs,
                };
                SelectionRequest::Budget { budget_seconds }
            };

            let plan = select_for_working_tree(&repo_root, request)?;
            let selected: Vec<McpSelectedTest> = plan
                .selected_plan
                .unwrap_or_default()
                .into_iter()
                .map(|s| McpSelectedTest {
                    test_id: s.test_id,
                    predicted_fail_prob: s.predicted_prob,
                    est_duration_ms: s.est_duration_ms,
                    reason: s.selection_reason.into(),
                })
                .collect();
            let estimated_duration_ms: u64 = selected.iter().map(|s| s.est_duration_ms).sum();

            Ok(SelectTestsOutput {
                model_mode: plan.model_mode.to_string(),
                diff_files_count: plan.diff.files.len(),
                total_discovered_tests: plan.discovered.len(),
                selected_count: selected.len(),
                estimated_duration_ms,
                selected,
            })
        })
        .await
        .map_err(|e| format!("blocking task panicked: {e}"))?
        .map_err(|e| format!("{e:#}"))?;

        Ok(Json(output))
    }

    /// Selects and executes the highest-value tests within the time budget, saves results
    /// to `.pts/store.db`, and returns structured outcomes with bounded tracebacks.
    #[tool(
        name = "run_fast_tests",
        description = "Select and execute the highest-value tests for the current working-tree diff, persist outcomes to .pts/store.db, and return structured pass/fail results."
    )]
    pub async fn run_fast_tests(
        &self,
        Parameters(params): Parameters<RunFastTestsParams>,
    ) -> Result<Json<RunFastTestsOutput>, String> {
        let repo_root = Arc::clone(&self.repo_root);
        let output = tokio::task::spawn_blocking(move || -> Result<RunFastTestsOutput> {
            let request = if params.run_all {
                SelectionRequest::All
            } else if let Some(top_k) = params.top_k {
                SelectionRequest::TopK { top_k }
            } else {
                let budget_seconds = match params.budget_seconds {
                    Some(secs) => secs,
                    None => Config::load(&repo_root)?.default_budget_secs,
                };
                SelectionRequest::Budget { budget_seconds }
            };

            let report = run_selected_tests(&repo_root, request)?;
            Ok(RunFastTestsOutput {
                run_id: report.run_id,
                model_mode: report.model_mode,
                passed: report.passed,
                failed: report.failed,
                errored: report.errored,
                skipped: report.skipped,
                total_duration_ms: report.total_duration_ms,
                failures: report.failures,
            })
        })
        .await
        .map_err(|e| format!("blocking task panicked: {e}"))?
        .map_err(|e| format!("{e:#}"))?;

        Ok(Json(output))
    }

    /// Inspects the exact 16 features and score behind a test's ranking for the current diff.
    #[tool(
        name = "explain_selection",
        description = "Explain a specific test's ranking by returning all 16 extracted features, active model mode, failure probability, yield score, and top contributing signals."
    )]
    pub async fn explain_selection(
        &self,
        Parameters(params): Parameters<ExplainSelectionParams>,
    ) -> Result<Json<ExplainReport>, String> {
        let repo_root = Arc::clone(&self.repo_root);
        let test_id = params.test_id;
        let report =
            tokio::task::spawn_blocking(move || explain_test(&repo_root, &test_id))
                .await
                .map_err(|e| format!("blocking task panicked: {e}"))?
                .map_err(|e| format!("{e:#}"))?;

        Ok(Json(report))
    }
}

#[tool_handler(
    router = self.tool_router,
    name = "pts",
    instructions = "Predictive Test Selector (PTS): predict, explain, and run the tests most likely to fail for the current working-tree git diff."
)]
impl ServerHandler for PtsMcpServer {}

/// Starts the PTS MCP server over stdio on a dedicated Tokio runtime.
pub fn run_mcp(repo_root: &Path) -> Result<()> {
    let root = repo_root.to_path_buf();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to initialize Tokio runtime for MCP server: {e}"))?;

    rt.block_on(async move {
        info!(repo = %root.display(), "starting PTS MCP stdio server");
        let server = PtsMcpServer::new(root);
        let running = server
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|e| anyhow!("MCP stdio handshake failed: {e}"))?;
        running
            .waiting()
            .await
            .map_err(|e| anyhow!("MCP stdio server terminated with error: {e}"))?;
        info!("PTS MCP stdio server shut down cleanly");
        Ok(())
    })
}
