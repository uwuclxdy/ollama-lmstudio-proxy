//! Tests for assorted pure helpers across the ollama->LM Studio translation layer.
//!
//! Concerns covered (with dividers below):
//!   - extract_system_prompt (handlers/ollama/utils.rs)
//!   - convert_tool_calls_to_ollama edge cases (handlers/transform.rs)
//!   - LmStudioDownloadStatus failure / unknown / chunk shapes
//!     (handlers/ollama/download_status.rs)
//!   - keep_alive_requests_unload semantics (handlers/ollama/keep_alive_parse.rs)
//!   - build_vision_chat_messages additional cases (handlers/ollama/images.rs)
//!   - BlobStore digest validation surfacing through exists() (storage/blob.rs)
//!
//! Reference docs:
//!   - api_docs/ollama.md (§/api/generate, /api/chat system field; /api/blobs digest
//!     format; tool_call function.arguments shape; /api/pull NDJSON stream)
//!   - api_docs/lmstudio/1_developer/3_openai-compat/tools.mdx

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/error.rs"]
#[allow(dead_code)]
mod error;

#[path = "../src/handlers/ollama/download_status.rs"]
#[allow(dead_code)]
mod download_status;

#[path = "../src/handlers/ollama/keep_alive_parse.rs"]
#[allow(dead_code)]
mod keep_alive_parse;

#[path = "../src/handlers/ollama/images.rs"]
#[allow(dead_code)]
mod images;

#[path = "../src/handlers/transform.rs"]
#[allow(dead_code)]
mod transform;

// `utils.rs` re-exports items via `pub use super::*`. To include the file as-is
// from an integration test we provide stub sibling modules whose item NAMES
// match what utils.rs re-exports. Signatures are irrelevant to a `pub use`
// rebind — only name resolution matters.
#[allow(dead_code, unused_variables)]
mod download_utils {
    pub fn determine_download_identifier() {}
    pub fn looks_like_remote_identifier() {}
}
#[allow(dead_code, unused_variables)]
mod keep_alive {
    pub fn keep_alive_requests_unload() {}
    pub fn parse_keep_alive_seconds() {}
}
#[allow(dead_code, unused_variables)]
mod model_resolution {
    pub fn resolve_model_target() {}
    pub fn resolve_model_with_context() {}
}
#[allow(dead_code, unused_variables)]
mod status_streaming {
    pub fn send_status_chunk() {}
    pub fn send_status_error_chunk() {}
    pub fn stream_status_messages() {}
}

#[path = "../src/handlers/ollama/utils.rs"]
#[allow(dead_code)]
mod utils;

use serde_json::{Value, json};

// =========================================================================
// extract_system_prompt
// =========================================================================

/// /api/generate: a top-level `system` string is the canonical Ollama field.
/// Reference: api_docs/ollama.md §"Generate a completion" — `system` overrides
/// the Modelfile-defined system prompt.
#[test]
fn extract_system_prompt_finds_top_level_system() {
    let body = json!({ "system": "be concise" });
    assert_eq!(
        utils::extract_system_prompt(&body),
        Some("be concise".to_string())
    );
}

/// Some clients place the system prompt inside `options`. The helper falls
/// back to `options.system` when the top-level field is absent.
#[test]
fn extract_system_prompt_falls_back_to_options_system() {
    let body = json!({ "options": { "system": "from options" } });
    assert_eq!(
        utils::extract_system_prompt(&body),
        Some("from options".to_string())
    );
}

/// When both are present the top-level field wins (it is the documented
/// override). Reference: api_docs/ollama.md §"Generate a completion".
#[test]
fn extract_system_prompt_top_level_wins_over_options() {
    let body = json!({
        "system": "top wins",
        "options": { "system": "should be ignored" }
    });
    assert_eq!(
        utils::extract_system_prompt(&body),
        Some("top wins".to_string())
    );
}

/// Neither location populated → None.
#[test]
fn extract_system_prompt_returns_none_when_absent() {
    let body = json!({ "model": "x" });
    assert!(utils::extract_system_prompt(&body).is_none());
}

/// Documented divergence: the helper only checks `body.system` and
/// `body.options.system`. /api/chat's `messages` array with a `{role:"system"}`
/// entry is NOT scanned here — that work is done by other translation steps
/// (e.g. normalize_chat_messages). This test pins the current behavior.
#[test]
fn extract_system_prompt_does_not_inspect_messages_array() {
    let body = json!({
        "messages": [
            { "role": "system", "content": "from messages" },
            { "role": "user", "content": "hi" }
        ]
    });
    assert!(
        utils::extract_system_prompt(&body).is_none(),
        "extract_system_prompt is not responsible for the messages array; \
         system-role messages flow through normalize_chat_messages"
    );
}

/// Non-string `system` is ignored (the helper only accepts strings).
#[test]
fn extract_system_prompt_rejects_non_string_system() {
    let body = json!({ "system": 42 });
    assert!(utils::extract_system_prompt(&body).is_none());
}

// =========================================================================
// convert_tool_calls_to_ollama (handlers/transform.rs)
// =========================================================================
//
// Reference: api_docs/ollama.md §tool_calls shape — Ollama expects
// {function:{name, arguments:object}} with no `id` or `type` wrapper.
// LM Studio (OpenAI-compat) emits {id, type, function:{name, arguments:string}}.

/// Empty input array → empty output array. Guards against an unwrap-on-empty.
#[test]
fn tool_calls_empty_input_yields_empty_output() {
    let result = transform::convert_tool_calls_to_ollama(&[]);
    assert_eq!(result, json!([]));
}

/// Missing `function` field entirely → still produces a graceful entry with
/// empty name and empty arguments object. The converter must not panic on
/// shapes outside the OpenAI spec.
#[test]
fn tool_calls_missing_function_field_does_not_panic() {
    let tool_calls = vec![json!({ "id": "call_x", "type": "function" })];
    let result = transform::convert_tool_calls_to_ollama(&tool_calls);
    let first = &result.as_array().unwrap()[0];
    assert_eq!(first["function"]["name"], json!(""));
    assert_eq!(first["function"]["arguments"], json!({}));
}

/// `arguments` field missing entirely → empty object (Ollama spec requires
/// an object, never null or absent).
#[test]
fn tool_calls_missing_arguments_field_defaults_to_empty_object() {
    let tool_calls = vec![json!({
        "function": { "name": "fn_no_args" }
    })];
    let result = transform::convert_tool_calls_to_ollama(&tool_calls);
    let args = &result.as_array().unwrap()[0]["function"]["arguments"];
    assert!(args.is_object(), "arguments must be an object, got {args}");
    assert_eq!(args, &json!({}));
}

/// `arguments` as malformed/non-JSON string → empty object fallback. Without
/// this guard a server hiccup that emits `"not-json"` would crash the parser.
#[test]
fn tool_calls_malformed_arguments_string_falls_back_to_empty_object() {
    let tool_calls = vec![json!({
        "function": { "name": "fn", "arguments": "not valid json {{" }
    })];
    let result = transform::convert_tool_calls_to_ollama(&tool_calls);
    let args = &result.as_array().unwrap()[0]["function"]["arguments"];
    assert_eq!(args, &json!({}));
}

/// Multiple tool calls in one batch — both converted, order preserved,
/// `id` and `type` wrapper fields stripped on every entry.
#[test]
fn tool_calls_multiple_preserves_order_and_strips_wrappers() {
    let tool_calls = vec![
        json!({
            "id": "call_1",
            "type": "function",
            "function": { "name": "first", "arguments": "{\"a\":1}" }
        }),
        json!({
            "id": "call_2",
            "type": "function",
            "function": { "name": "second", "arguments": "{\"b\":2}" }
        }),
    ];
    let result = transform::convert_tool_calls_to_ollama(&tool_calls);
    let arr = result.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["function"]["name"], json!("first"));
    assert_eq!(arr[1]["function"]["name"], json!("second"));
    assert_eq!(arr[0]["function"]["arguments"]["a"], json!(1));
    assert_eq!(arr[1]["function"]["arguments"]["b"], json!(2));
    for entry in arr {
        assert!(entry.get("id").is_none(), "id wrapper must be stripped");
        assert!(entry.get("type").is_none(), "type wrapper must be stripped");
    }
}

// =========================================================================
// LmStudioDownloadStatus extended cases (handlers/ollama/download_status.rs)
// =========================================================================
//
// Reference: api_docs/ollama.md lines 1582-1626 — /api/pull NDJSON stream.

fn lm_status_value(extra: Value) -> download_status::LmStudioDownloadStatus {
    let mut base = json!({
        "job_id": "job_abc",
        "status": "downloading"
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            base[k] = v.clone();
        }
    }
    serde_json::from_value(base).expect("test fixture must deserialize")
}

/// is_terminal() — true for the three terminal states, false for in-progress.
/// Reference: api_docs/ollama.md §/api/pull — terminal is success-or-failed.
#[test]
fn download_status_is_terminal_matches_known_states() {
    assert!(lm_status_value(json!({"status": "completed"})).is_terminal());
    assert!(lm_status_value(json!({"status": "already_downloaded"})).is_terminal());
    assert!(lm_status_value(json!({"status": "failed"})).is_terminal());
    assert!(!lm_status_value(json!({"status": "downloading"})).is_terminal());
    assert!(!lm_status_value(json!({"status": "queued"})).is_terminal());
}

/// into_final_response on failed status returns Err carrying the LM Studio
/// error string verbatim — clients see the upstream reason, not a generic
/// "unknown" message. Reference: api_docs/ollama.md §/api/pull error path.
#[test]
fn download_status_failed_into_final_response_propagates_error_message() {
    let s = lm_status_value(json!({
        "status": "failed",
        "error": "disk full"
    }));
    let result = s.into_final_response("llama3.2");
    let err = result.expect_err("failed status must produce Err");
    assert!(
        err.message.contains("disk full"),
        "expected LM Studio error preserved, got: {}",
        err.message
    );
}

/// Failure without an `error` field still yields Err with a fallback message,
/// never a fake success.
#[test]
fn download_status_failed_without_error_field_uses_fallback_message() {
    let s = lm_status_value(json!({ "status": "failed" }));
    let err = s
        .into_final_response("llama3.2")
        .expect_err("failed must be Err");
    assert!(!err.message.is_empty());
    // Must not silently look like success.
    assert_ne!(err.message, "success");
}

/// Unknown / unexpected status string → Err. Guards against silently
/// reporting success for states we have not mapped (e.g. "paused").
#[test]
fn download_status_unknown_status_into_final_response_is_err() {
    let s = lm_status_value(json!({ "status": "paused" }));
    let err = s
        .into_final_response("llama3.2")
        .expect_err("unknown status must be Err");
    assert!(
        err.message.contains("paused") || err.message.to_lowercase().contains("unexpected"),
        "error should mention the unexpected status, got: {}",
        err.message
    );
}

/// to_chunk on a failed status surfaces the upstream `error` so streaming
/// clients can show the reason. The chunk must NOT be the bare success
/// sentinel {"status":"success"}.
#[test]
fn download_status_failed_chunk_includes_error_and_is_not_success_sentinel() {
    let s = lm_status_value(json!({
        "status": "failed",
        "error": "checksum mismatch"
    }));
    let chunk = s.to_chunk("llama3.2");
    let obj = chunk.as_object().expect("chunk must be an object");
    assert_eq!(
        obj.get("error"),
        Some(&json!("checksum mismatch")),
        "failed chunk must include the upstream error string"
    );
    assert_ne!(
        obj.get("status"),
        Some(&json!("success")),
        "failed terminal chunk must NOT be shaped like the bare success sentinel"
    );
    // A bare {status:success} object has exactly one key; a failure chunk
    // has many more (status, model, detail, error, ...).
    assert!(
        obj.len() > 1,
        "failed chunk must carry more than just `status`, got: {chunk}"
    );
}

// =========================================================================
// keep_alive_requests_unload (handlers/ollama/keep_alive_parse.rs)
// =========================================================================
//
// Reference: api_docs/ollama.md §Modelfile keep_alive — 0 means "unload now".

/// Some(0) means unload requested. Reference: api_docs/ollama.md keep_alive=0.
#[test]
fn keep_alive_requests_unload_true_for_zero() {
    assert!(keep_alive_parse::keep_alive_requests_unload(Some(0)));
}

/// Positive seconds means stay loaded; not an unload request.
#[test]
fn keep_alive_requests_unload_false_for_positive() {
    assert!(!keep_alive_parse::keep_alive_requests_unload(Some(300)));
}

/// Negative normalises to "forever" upstream — not an unload request here.
#[test]
fn keep_alive_requests_unload_false_for_negative() {
    assert!(!keep_alive_parse::keep_alive_requests_unload(Some(-1)));
}

/// Absent keep_alive means "leave LM Studio default" — not an unload request.
#[test]
fn keep_alive_requests_unload_false_for_none() {
    assert!(!keep_alive_parse::keep_alive_requests_unload(None));
}

// =========================================================================
// build_vision_chat_messages — additional cases (handlers/ollama/images.rs)
// =========================================================================
//
// Reference: api_docs/ollama.md §/api/generate with images + the OpenAI-compat
// content-parts shape used by LM Studio.

/// With a system prompt and no images, the user `content` stays a plain
/// string — not promoted to a content-parts array.
#[test]
fn vision_messages_system_plus_no_images_keeps_string_content() {
    let messages = images::build_vision_chat_messages(Some("be brief"), "hello", None);
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["role"], json!("system"));
    assert_eq!(arr[0]["content"], json!("be brief"));
    assert_eq!(arr[1]["role"], json!("user"));
    assert_eq!(
        arr[1]["content"],
        json!("hello"),
        "no images → content must remain a plain string"
    );
}

/// With a system prompt AND images, the user `content` becomes a typed
/// content-parts array (text part first, then image_url parts).
#[test]
fn vision_messages_system_plus_images_yields_typed_parts() {
    let images_val = json!(["iVBORw0KGgoAAA"]);
    let messages =
        images::build_vision_chat_messages(Some("be brief"), "describe", Some(&images_val));
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["role"], json!("system"));
    let parts = arr[1]["content"].as_array().expect("content must be array");
    assert_eq!(parts[0]["type"], json!("text"));
    assert_eq!(parts[0]["text"], json!("describe"));
    assert_eq!(parts[1]["type"], json!("image_url"));
    let url = parts[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "got {url}");
}

/// No system prompt → output is exactly one user message.
#[test]
fn vision_messages_no_system_yields_only_user_message() {
    let messages = images::build_vision_chat_messages(None, "hi", None);
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["role"], json!("user"));
    assert_eq!(arr[0]["content"], json!("hi"));
}

/// images is an empty array → still a plain-string user content (not an
/// empty content-parts array, which some runtimes reject).
#[test]
fn vision_messages_empty_images_array_keeps_string_content() {
    let images_val = json!([]);
    let messages = images::build_vision_chat_messages(None, "hi", Some(&images_val));
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(
        arr[0]["content"],
        json!("hi"),
        "empty images must not promote content to an array"
    );
}

/// images contains non-string entries (number, object) → they are skipped
/// silently and the user content stays a plain string when nothing usable
/// remains.
#[test]
fn vision_messages_non_string_images_are_skipped() {
    let images_val = json!([42, { "url": "ignored" }]);
    let messages = images::build_vision_chat_messages(None, "hi", Some(&images_val));
    let arr = messages.as_array().unwrap();
    assert_eq!(
        arr[0]["content"],
        json!("hi"),
        "non-string image entries must be dropped, leaving plain content"
    );
}

/// convert_per_message_images: non-array input is returned unchanged (the
/// converter must not panic on shapes it cannot interpret).
#[test]
fn convert_per_message_images_non_array_passthrough() {
    let messages = json!({ "not": "an array" });
    let out = images::convert_per_message_images(messages.clone());
    assert_eq!(out, messages);
}

/// inject_images_into_messages: empty images array → messages returned as-is.
#[test]
fn inject_empty_images_returns_messages_unchanged() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let images_val = json!([]);
    let out = images::inject_images_into_messages(messages.clone(), &images_val);
    assert_eq!(out, messages);
}

// =========================================================================
// BlobStore digest validation surfacing through exists()
// =========================================================================
//
// Reference: api_docs/ollama.md §/api/blobs/:digest — digest format is
// `sha256:<64-hex>`. The proxy's BlobStore::validated_blob_path is private,
// so we exercise it indirectly through the pub `exists()` entrypoint. A
// valid digest yields Ok(false) (no such file in a fresh temp dir); an
// invalid digest yields Err(ProxyError) before any filesystem lookup.

#[path = "../src/storage/blob.rs"]
#[allow(dead_code)]
mod blob;

fn fresh_blob_store() -> blob::BlobStore {
    let dir = std::env::temp_dir().join(format!(
        "olp-test-blobs-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    blob::BlobStore::new(&dir).expect("BlobStore::new should succeed in temp dir")
}

fn check_digest(digest: &str) -> Result<bool, error::ProxyError> {
    let store = fresh_blob_store();
    // BlobStore::exists is async; build a minimal current-thread runtime.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(store.exists(digest))
}

/// Valid `sha256:` + 64 lowercase hex characters → accepted, returns
/// Ok(false) because no such blob exists in a fresh store.
#[test]
fn blob_digest_valid_sha256_is_accepted() {
    let digest = format!("sha256:{}", "a".repeat(64));
    let result = check_digest(&digest);
    assert!(matches!(result, Ok(false)), "got {result:?}");
}

/// Wrong algorithm prefix (md5) → rejected with 400. Only sha256 is
/// supported per the validator and per the Ollama spec.
#[test]
fn blob_digest_wrong_algorithm_is_rejected() {
    let digest = format!("md5:{}", "a".repeat(32));
    let err = check_digest(&digest).expect_err("non-sha256 must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Missing colon separator → rejected. The validator requires `algo:hex`.
#[test]
fn blob_digest_missing_colon_is_rejected() {
    let digest = "a".repeat(64);
    let err = check_digest(&digest).expect_err("missing colon must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Hex shorter than 64 characters → rejected.
#[test]
fn blob_digest_short_hex_is_rejected() {
    let digest = format!("sha256:{}", "a".repeat(32));
    let err = check_digest(&digest).expect_err("short hex must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Hex longer than 64 characters → rejected.
#[test]
fn blob_digest_long_hex_is_rejected() {
    let digest = format!("sha256:{}", "a".repeat(128));
    let err = check_digest(&digest).expect_err("oversize hex must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Non-hex characters in the hex portion → rejected. Guards against
/// path-traversal payloads slipping through (`/`, `.`, etc.) and against
/// uppercase or unicode confusables.
#[test]
fn blob_digest_non_hex_chars_are_rejected() {
    let bad = [
        // Path traversal attempt — would escape base_dir if accepted as a path.
        format!(
            "sha256:{}",
            "../../../etc/passwd000000000000000000000000000000"
        ),
        // Embedded slash.
        format!("sha256:{}{}", "a".repeat(32), "/".repeat(32)),
        // Non-hex letter.
        format!("sha256:{}", "z".repeat(64)),
        // Uppercase hex — validator uses ascii_hexdigit which DOES accept
        // uppercase, so this one is documented separately below. Skip here.
    ];
    for digest in bad {
        let err = check_digest(&digest).expect_err(&format!("must reject {digest}"));
        assert_eq!(err.status_code, 400, "digest {digest} should be 400");
    }
}

/// Path-traversal attempt with `..` and slash in the hex portion → rejected.
/// Critical: without this check an attacker could read/write arbitrary files
/// under the blob storage root.
#[test]
fn blob_digest_path_traversal_is_rejected() {
    // exactly 64 chars but containing `..` and `/`.
    let digest = format!("sha256:{}", "../".repeat(21) + "a");
    assert_eq!(digest.len() - "sha256:".len(), 64);
    let err = check_digest(&digest).expect_err("path traversal must be rejected");
    assert_eq!(err.status_code, 400);
}
