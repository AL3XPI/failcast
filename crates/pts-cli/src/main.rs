use std::process::ExitCode;

use clap::Parser;
use pts_cli::cli::{self, Cli};
use tracing_subscriber::EnvFilter;

fn main() -> ExitCode {
    // Route all tracing logs to stderr so stdout stays clean for JSON and MCP framing.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let parsed = Cli::parse();
    let cwd = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("pts error: failed to determine current directory: {e}");
            return ExitCode::from(2);
        }
    };

    match cli::execute(parsed, &cwd) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(err) => {
            eprintln!("pts error: {err:#}");
            ExitCode::from(2)
        }
    }
}
