//! Per-event mapping for LM Studio's native `/api/v1/chat` SSE stream.
//!
//! The native stream uses named SSE events (`event: <type>\ndata: <json>\n\n`),
//! unlike the OpenAI-compat path's bare `data:` lines. This module owns the
//! per-event semantics only: given an already-split `(event_type, data)`, it
//! produces a [`ChoiceDeltaPayload`] (reusing the OpenAI path's delta shape) or
//! signals end/error. Byte framing belongs to the caller, but a small
//! [`parse_native_sse_message`] helper is provided so the framing layer can
//! reuse one consistent parse.
//!
//! Source of truth:
//! - `api-docs/lmstudio/1_developer/2_rest/streaming-events.md`

use serde_json::Value;

use crate::lmstudio::native_chat::native_tool_call_to_openai;
use crate::streaming::chunks::{ChoiceDeltaPayload, ChunkProcessingState};

/// Outcome of mapping a single native SSE event.
///
/// `Delta` carries content/thinking to emit as an intermediate Ollama chunk.
/// `End` signals `chat.end` and hands the caller the final `result` block (for
/// stats extraction). `Error` surfaces the native `error` payload; per the
/// streaming-events doc it is recoverable — `chat.end` still follows with
/// whatever was generated, so the caller should surface it and keep reading.
/// `ModelLoaded` carries `model_load.end`'s `load_time_seconds` for
/// `load_duration` when the final stats block omits it. `Ignore` covers
/// boundary/progress events and tool-call fragments (those accumulate into
/// `state` for the stream driver's single pre-final emission).
pub enum NativeEvent {
    Delta(ChoiceDeltaPayload),
    End(NativeChatEnd),
    Error(NativeStreamError),
    ModelLoaded(f64),
    Ignore,
}

/// Data extracted from a `chat.end` event for building the final timing chunk.
///
/// `result` is the aggregated response object (same shape as a non-streaming
/// `/api/v1/chat` body); `stats` is pulled out for convenience. The native API
/// exposes no finish-reason anywhere, so the caller derives `done_reason` via
/// the shared proxy heuristic (`native_done_reason_heuristic`).
pub struct NativeChatEnd {
    pub result: Value,
    pub stats: Option<Value>,
}

/// A native `error` event payload, surfaced so the caller can fail the stream.
pub struct NativeStreamError {
    pub error_type: String,
    pub message: String,
    pub code: Option<String>,
    pub param: Option<String>,
}

impl NativeStreamError {
    /// Render a single human-readable line for an error chunk / log.
    pub fn to_message(&self) -> String {
        let mut out = format!("{}: {}", self.error_type, self.message);
        if let Some(code) = &self.code {
            out.push_str(&format!(" (code: {code})"));
        }
        if let Some(param) = &self.param {
            out.push_str(&format!(" (param: {param})"));
        }
        out
    }
}

/// Map one native SSE event (`event_type` + parsed `data`) to a [`NativeEvent`].
///
/// Tool-call argument/success events accumulate into `state` (reusing the
/// OpenAI accumulator) without producing a client-visible delta; the stream
/// driver emits the assembled calls once, before the final chunk. Boundary,
/// progress and start/end-marker events return [`NativeEvent::Ignore`].
pub fn map_native_event(
    event_type: &str,
    data: &Value,
    state: &mut ChunkProcessingState,
) -> NativeEvent {
    match event_type {
        "reasoning.delta" => {
            let thinking = data
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string();
            NativeEvent::Delta(ChoiceDeltaPayload {
                content: String::new(),
                thinking,
            })
        }
        "message.delta" => {
            let content = data
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string();
            NativeEvent::Delta(ChoiceDeltaPayload {
                content,
                thinking: String::new(),
            })
        }
        "tool_call.start" => {
            // Native events carry no call index; calls are strictly sequential,
            // so each start boundary opens the next accumulator slot. Without
            // this, parallel calls collapse into slot 0 (last one wins).
            state.begin_native_tool_call();
            NativeEvent::Ignore
        }
        "tool_call.arguments" | "tool_call.success" => {
            // Reshape the native tool entry to the OpenAI-ish shape the
            // accumulator expects; emission happens once, pre-final. Successive
            // `arguments` payloads are snapshots, so replacing per slot is the
            // correct merge (`tool_call.success` carries the final object).
            let openai_shaped = native_tool_call_to_openai(data);
            state.accumulate_native_tool_call(openai_shaped);
            NativeEvent::Ignore
        }
        "error" => NativeEvent::Error(parse_native_error(data)),
        "model_load.end" => match data.get("load_time_seconds").and_then(|v| v.as_f64()) {
            Some(seconds) => NativeEvent::ModelLoaded(seconds),
            None => NativeEvent::Ignore,
        },
        "chat.end" => NativeEvent::End(parse_chat_end(data)),
        // chat.start, model_load.start/progress, prompt_processing.*,
        // reasoning.start/end, message.start/end, tool_call.failure (boundary only).
        _ => NativeEvent::Ignore,
    }
}

/// Extract the `error` block from a native `error` event into a typed struct.
fn parse_native_error(data: &Value) -> NativeStreamError {
    let error = data.get("error").unwrap_or(data);
    NativeStreamError {
        error_type: error
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        message: error
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        code: error
            .get("code")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
        param: error
            .get("param")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
    }
}

/// Extract `result.stats` from a `chat.end` event.
///
/// The `result` object is cloned out so a caller can run it through the
/// non-streaming converter if it wants the full aggregated body. No
/// finish-reason exists to extract (neither `chat.end` nor the stats block
/// carry one) — `done_reason` is the caller's heuristic.
pub fn parse_chat_end(data: &Value) -> NativeChatEnd {
    let result = data.get("result").cloned().unwrap_or(Value::Null);
    let stats = result.get("stats").cloned();

    NativeChatEnd { result, stats }
}

/// Parse one raw native SSE message block into `(event_type, data)`.
///
/// A block has the wire form `event: <type>\ndata: <json>` (lines may carry a
/// trailing `\r`). Returns `None` when no `event:` line is present or the
/// `data:` payload is missing or not valid JSON. Multi-line `data:` fields are
/// concatenated per the SSE spec.
pub fn parse_native_sse_message(block: &str) -> Option<(String, Value)> {
    let mut event_type: Option<String> = None;
    let mut data = String::new();
    let mut saw_data = false;

    for line in block.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(rest) = line.strip_prefix("event:") {
            event_type = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            if saw_data {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            saw_data = true;
        }
    }

    let event_type = event_type?;
    if !saw_data {
        return None;
    }
    let parsed = serde_json::from_str(&data).ok()?;
    Some((event_type, parsed))
}

#[cfg(test)]
#[path = "../../tests/unit/streaming_native.rs"]
mod tests;
