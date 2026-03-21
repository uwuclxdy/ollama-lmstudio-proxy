use std::borrow::Cow;

use serde_json::{Value, json};

pub struct RequestBuilder {
    body: serde_json::Map<String, Value>,
}

impl RequestBuilder {
    pub fn new() -> Self {
        Self {
            body: serde_json::Map::new(),
        }
    }

    pub fn add_required<T: Into<Value>>(mut self, key: &str, value: T) -> Self {
        self.body.insert(key.to_string(), value.into());
        self
    }

    pub fn build(self) -> Value {
        Value::Object(self.body)
    }
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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

pub struct PreparedBody {
    pub bytes: Option<Vec<u8>>,
    pub is_json: bool,
}

pub fn prepare_request_body(
    json_body: Option<Value>,
    original_bytes: &[u8],
) -> Result<PreparedBody, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(value) = json_body {
        let serialized = serde_json::to_vec(&value)
            .map_err(|e| format!("failed to serialize request body: {}", e))?;
        Ok(PreparedBody {
            bytes: Some(serialized),
            is_json: true,
        })
    } else if !original_bytes.is_empty() {
        Ok(PreparedBody {
            bytes: Some(original_bytes.to_vec()),
            is_json: false,
        })
    } else {
        Ok(PreparedBody {
            bytes: None,
            is_json: false,
        })
    }
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

fn apply_top_level_params(top: &TopLevelParams<'_>, request_obj: &mut serde_json::Map<String, Value>) {
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
    let mut builder = RequestBuilder::new().add_required("model", model_lm_studio_id);

    match request_type {
        LMStudioRequestType::Chat { messages, stream } => {
            builder = builder
                .add_required("messages", messages.clone())
                .add_required("stream", stream);
            if let Some(tools_val) = ollama_tools
                && tools_val.is_array()
                && !tools_val.as_array().unwrap().is_empty()
            {
                builder = builder.add_required("tools", tools_val.clone());
            }
        }
        LMStudioRequestType::Completion { prompt, stream } => {
            builder = builder
                .add_required("prompt", prompt.as_ref())
                .add_required("stream", stream);
        }
        LMStudioRequestType::Embeddings { input } => {
            builder = builder.add_required("input", input.clone());
        }
    }

    let lm_studio_mapped_params = map_ollama_to_lmstudio_params(ollama_options, structured_format);
    let mut request_json = builder.build();

    if let Some(request_obj) = request_json.as_object_mut() {
        for (key, value) in lm_studio_mapped_params {
            request_obj.insert(key, value);
        }
    }

    if let (Some(top), Some(request_obj)) = (top_level, request_json.as_object_mut()) {
        apply_top_level_params(top, request_obj);
    }

    request_json
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

        if let Some(system) = options.get("system") {
            params.insert("system".to_string(), system.clone());
        }
    }
}

fn map_token_limits(ollama_options: Option<&Value>, params: &mut serde_json::Map<String, Value>) {
    if let Some(options) = ollama_options
        && let Some(max_tokens) = options
            .get("max_tokens")
            .or_else(|| options.get("num_predict"))
    {
        params.insert("max_tokens".to_string(), max_tokens.clone());
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
    "num_ctx",
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
        log::debug!("unsupported options ignored: {}", keys.join(", "));
    }
}

fn convert_structured_format(format_value: &Value) -> Option<Value> {
    match format_value {
        Value::String(mode) if mode.eq_ignore_ascii_case("json") => {
            Some(json!({ "type": "json_object" }))
        }
        Value::String(mode) if mode.eq_ignore_ascii_case("text") => Some(json!({ "type": "text" })),
        Value::Object(_) => Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "ollama_format",
                "strict": true,
                "schema": format_value.clone()
            }
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn forwards_presence_penalty() {
        let options = json!({ "presence_penalty": 0.5 });
        let params = map_ollama_to_lmstudio_params(Some(&options), None);
        assert_eq!(params.get("presence_penalty"), Some(&json!(0.5)));
    }

    #[test]
    fn forwards_frequency_penalty() {
        let options = json!({ "frequency_penalty": 0.3 });
        let params = map_ollama_to_lmstudio_params(Some(&options), None);
        assert_eq!(params.get("frequency_penalty"), Some(&json!(0.3)));
    }

    #[test]
    fn forwards_min_p() {
        let options = json!({ "min_p": 0.05 });
        let params = map_ollama_to_lmstudio_params(Some(&options), None);
        assert_eq!(params.get("min_p"), Some(&json!(0.05)));
    }

    #[test]
    fn collects_unsupported_options() {
        let options = json!({ "num_ctx": 4096, "mirostat": 1, "temperature": 0.7 });
        let unsupported = collect_unsupported_keys(&options);
        assert!(unsupported.contains(&"num_ctx"), "expected num_ctx in {:?}", unsupported);
        assert!(unsupported.contains(&"mirostat"), "expected mirostat in {:?}", unsupported);
        // temperature is supported — must NOT appear
        assert!(!unsupported.contains(&"temperature"), "temperature should not appear in {:?}", unsupported);
    }

    #[test]
    fn log_unsupported_options_does_not_panic() {
        // Smoke test: calling with unsupported keys must not panic (log output not asserted)
        let options = json!({ "num_ctx": 4096, "mirostat": 1 });
        log_unsupported_options(&options);

        // No keys: also must not panic
        let empty = json!({ "temperature": 0.7 });
        log_unsupported_options(&empty);
    }

    #[test]
    fn top_level_params_think_true_emits_reasoning_on() {
        let think_val = json!(true);
        let top = TopLevelParams {
            think: Some(&think_val),
            logprobs: None,
            top_logprobs: None,
        };
        let request = build_lm_studio_request(
            "mymodel",
            LMStudioRequestType::Completion {
                prompt: std::borrow::Cow::Borrowed("hello"),
                stream: false,
            },
            None,
            None,
            None,
            Some(&top),
        );
        assert_eq!(request.get("reasoning"), Some(&json!("on")));
    }

    #[test]
    fn top_level_params_think_false_emits_reasoning_off() {
        let think_val = json!(false);
        let top = TopLevelParams {
            think: Some(&think_val),
            logprobs: None,
            top_logprobs: None,
        };
        let request = build_lm_studio_request(
            "mymodel",
            LMStudioRequestType::Completion {
                prompt: std::borrow::Cow::Borrowed("hello"),
                stream: false,
            },
            None,
            None,
            None,
            Some(&top),
        );
        assert_eq!(request.get("reasoning"), Some(&json!("off")));
    }

    #[test]
    fn top_level_params_think_string_passes_through() {
        let think_val = json!("high");
        let top = TopLevelParams {
            think: Some(&think_val),
            logprobs: None,
            top_logprobs: None,
        };
        let request = build_lm_studio_request(
            "mymodel",
            LMStudioRequestType::Completion {
                prompt: std::borrow::Cow::Borrowed("hello"),
                stream: false,
            },
            None,
            None,
            None,
            Some(&top),
        );
        assert_eq!(request.get("reasoning"), Some(&json!("high")));
    }

    #[test]
    fn top_level_params_absent_think_emits_no_reasoning() {
        let top = TopLevelParams { think: None, logprobs: None, top_logprobs: None };
        let request = build_lm_studio_request(
            "mymodel",
            LMStudioRequestType::Completion {
                prompt: std::borrow::Cow::Borrowed("hello"),
                stream: false,
            },
            None,
            None,
            None,
            Some(&top),
        );
        assert!(request.get("reasoning").is_none());
    }

    #[test]
    fn top_level_params_logprobs_forwarded() {
        let lp = json!(true);
        let top = TopLevelParams { think: None, logprobs: Some(&lp), top_logprobs: None };
        let request = build_lm_studio_request(
            "mymodel",
            LMStudioRequestType::Completion {
                prompt: std::borrow::Cow::Borrowed("hello"),
                stream: false,
            },
            None,
            None,
            None,
            Some(&top),
        );
        assert_eq!(request.get("logprobs"), Some(&json!(true)));
    }

    #[test]
    fn top_level_params_work_on_chat_type_too() {
        // think applies to both chat and generate paths
        let think_val = json!("medium");
        let top = TopLevelParams { think: Some(&think_val), logprobs: None, top_logprobs: None };
        let messages = json!([{"role": "user", "content": "hi"}]);
        let request = build_lm_studio_request(
            "mymodel",
            LMStudioRequestType::Chat { messages: &messages, stream: false },
            None,
            None,
            None,
            Some(&top),
        );
        assert_eq!(request.get("reasoning"), Some(&json!("medium")));
    }
}
