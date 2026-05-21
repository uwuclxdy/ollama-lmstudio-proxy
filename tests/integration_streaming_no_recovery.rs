// Separate test binary for streaming behaviour when `enable_chunk_recovery` is
// OFF. Lives in its own binary because `RUNTIME_CONFIG` is a process-global
// `OnceLock`; the main integration binary fixes it to recovery=true, so the
// recovery=false path has to be exercised here.

#[path = "common/mod.rs"]
mod common;

#[path = "integration/streaming_no_recovery.rs"]
mod streaming_no_recovery;
