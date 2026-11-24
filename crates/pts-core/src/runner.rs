use std::path::Path;

use crate::{PtsError, RunnerResult, TestCase};

pub trait TestRunner: Send + Sync {
    fn name(&self) -> &'static str;
    fn discover(&self, repo_root: &Path) -> Result<Vec<TestCase>, PtsError>;
    fn execute(
        &self,
        repo_root: &Path,
        selected_ids: Option<&[String]>,
    ) -> Result<RunnerResult, PtsError>;
}
