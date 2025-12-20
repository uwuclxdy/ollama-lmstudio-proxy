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

    params
}

pub fn build_lm_studio_request(
    model_lm_studio_id: &str,
    request_type: LMStudioRequestType,
    ollama_options: Option<&Value>,
    ollama_tools: Option<&Value>,
    structured_format: Option<&Value>,
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
