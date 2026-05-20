use std::borrow::Cow;

use serde_json::{Value, json};

pub enum LMStudioRequestType<'a> {
    Chat { messages: &'a Value, stream: bool },
    Completion { prompt: Cow<'a, str>, stream: bool },
    Embeddings { input: &'a Value },
}

pub struct TopLevelParams<'a> {
    pub think: Option<&'a Value>,
    pub logprobs: Option<&'a Value>,
    pub top_logprobs: Option<&'a Value>,
}

pub fn map_ollama_to_lmstudio_params(
    ollama_options: Option<&Value>,
    structured_format: Option<&Value>,
) -> serde_json::Map<String, Value> {
    let mut params = serde_json::Map::new();

    map_direct_params(ollama_options, &mut params);
    map_token_limits(ollama_options, &mut params);
    map_penalties(ollama_options, &mut params);
    map_format_params(ollama_options, structured_format, &mut params);

    if let Some(options) = ollama_options {
        log_unsupported_options(options);
    }

    params
}

fn apply_top_level_params(
    top: &TopLevelParams<'_>,
    request_obj: &mut serde_json::Map<String, Value>,
) {
    if let Some(think_val) = top.think {
        let reasoning: Value = match think_val {
            Value::Bool(true) => json!("on"),
            Value::Bool(false) => json!("off"),
            Value::String(s) => json!(s),
            other => {
                log::debug!("think: unexpected value type {:?}, forwarding as-is", other);
                other.clone()
            }
        };
        request_obj.insert("reasoning".to_string(), reasoning);
    }
    if let Some(lp) = top.logprobs {
        request_obj.insert("logprobs".to_string(), lp.clone());
    }
    if let Some(tlp) = top.top_logprobs {
        request_obj.insert("top_logprobs".to_string(), tlp.clone());
    }
}

pub fn build_lm_studio_request(
    model_lm_studio_id: &str,
    request_type: LMStudioRequestType,
    ollama_options: Option<&Value>,
    ollama_tools: Option<&Value>,
    structured_format: Option<&Value>,
    top_level: Option<&TopLevelParams<'_>>,
) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(model_lm_studio_id));

    match request_type {
        LMStudioRequestType::Chat { messages, stream } => {
            body.insert("messages".to_string(), messages.clone());
            body.insert("stream".to_string(), json!(stream));
            if let Some(tools_val) = ollama_tools
                && let Some(tools_arr) = tools_val.as_array()
                && !tools_arr.is_empty()
            {
                body.insert("tools".to_string(), tools_val.clone());
            }
        }
        LMStudioRequestType::Completion { prompt, stream } => {
            body.insert("prompt".to_string(), json!(prompt.as_ref()));
            body.insert("stream".to_string(), json!(stream));
        }
        LMStudioRequestType::Embeddings { input } => {
            body.insert("input".to_string(), input.clone());
        }
    }

    for (key, value) in map_ollama_to_lmstudio_params(ollama_options, structured_format) {
        body.insert(key, value);
    }

    if let Some(top) = top_level {
        apply_top_level_params(top, &mut body);
    }

    Value::Object(body)
}

fn map_direct_params(ollama_options: Option<&Value>, params: &mut serde_json::Map<String, Value>) {
    if let Some(options) = ollama_options {
        const DIRECT_MAPPINGS: &[&str] = &[
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "stop",
            "truncate",
            "dimensions",
            "presence_penalty",
            "frequency_penalty",
            "min_p",
        ];

        for param in DIRECT_MAPPINGS {
            if let Some(value) = options.get(param) {
                params.insert(param.to_string(), value.clone());
            }
        }

        if let Some(logit_bias) = options.get("logit_bias") {
            params.insert("logit_bias".to_string(), logit_bias.clone());
        }
    }
}

fn map_token_limits(ollama_options: Option<&Value>, params: &mut serde_json::Map<String, Value>) {
    let Some(options) = ollama_options else {
        return;
    };

    if let Some(max_tokens) = options
        .get("max_tokens")
        .or_else(|| options.get("num_predict"))
    {
        params.insert("max_tokens".to_string(), max_tokens.clone());
    }

    if let Some(ctx) = options.get("num_ctx") {
        params.insert("context_length".to_string(), ctx.clone());
    }
}

fn map_penalties(ollama_options: Option<&Value>, params: &mut serde_json::Map<String, Value>) {
    if let Some(options) = ollama_options
        && let Some(repeat_penalty_val) = options.get("repeat_penalty")
    {
        if !params.contains_key("frequency_penalty") && !params.contains_key("presence_penalty") {
            params.insert("repeat_penalty".to_string(), repeat_penalty_val.clone());
        } else if !params.contains_key("frequency_penalty") {
            params.insert("frequency_penalty".to_string(), repeat_penalty_val.clone());
        }
    }
}

fn map_format_params(
    ollama_options: Option<&Value>,
    structured_format: Option<&Value>,
    params: &mut serde_json::Map<String, Value>,
) {
    let format_source =
        structured_format.or_else(|| ollama_options.and_then(|options| options.get("format")));
    if let Some(format_value) = format_source
        && let Some(converted) = convert_structured_format(format_value)
    {
        params.insert("response_format".to_string(), converted);
    }
}

const UNSUPPORTED_OPTION_KEYS: &[&str] = &[
    "template",
    "repeat_last_n",
    "tfs_z",
    "typical_p",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "penalize_newline",
    "num_keep",
    "num_batch",
    "num_gpu",
    "num_thread",
    "numa",
    "use_mmap",
    "use_mlock",
    "vocab_only",
];

pub(crate) fn collect_unsupported_keys(options: &Value) -> Vec<&'static str> {
    UNSUPPORTED_OPTION_KEYS
        .iter()
        .copied()
        .filter(|key| options.get(key).is_some())
        .collect()
}

pub(crate) fn log_unsupported_options(options: &Value) {
    let keys = collect_unsupported_keys(options);
    if !keys.is_empty() {
        log::warn!(
            "Ollama options ignored (LM Studio does not support them): {}",
            keys.join(", ")
        );
    }
}

fn convert_structured_format(format_value: &Value) -> Option<Value> {
    match format_value {
        Value::String(mode) if mode.eq_ignore_ascii_case("json") => Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "json",
                "schema": { "type": "object" }
            }
        })),
        Value::String(mode) if mode.eq_ignore_ascii_case("text") => Some(json!({ "type": "text" })),
        Value::Object(_) => Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "ollama_format",
                "strict": "true",
                "schema": format_value.clone()
            }
        })),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_request.rs"]
mod tests;
