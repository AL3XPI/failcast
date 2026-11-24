use thiserror::Error;

#[derive(Debug, Error)]
pub enum PtsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("runner error: {0}")]
    Runner(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("PTS is not initialized in this repository; run `pts init` first")]
    NotInitialized,

    #[error("not yet implemented: {0}")]
    Unimplemented(String),
}
