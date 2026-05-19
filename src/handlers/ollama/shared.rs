//! Cross-handler helpers (chat + generate share these).

use serde_json::Value;

use crate::http::request::TopLevelParams;

/// Pull the three Ollama top-level forwarded keys (`think`, `logprobs`,
/// `top_logprobs`) out of a request body. Both /api/chat and /api/generate
/// forward these to LM Studio (`reasoning`, `logprobs`, `top_logprobs`).
pub fn make_top_level_params(body: &Value) -> TopLevelParams<'_> {
    TopLevelParams {
        think: body.get("think"),
        logprobs: body.get("logprobs"),
        top_logprobs: body.get("top_logprobs"),
    }
}
