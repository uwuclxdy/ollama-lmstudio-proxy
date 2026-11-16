use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use base64::Engine as _;
use serde::de::{self, Visitor};
use serde::ser::{Error as _, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use serde_json::{Value, json};

use crate::http::json_as_i64;

pub enum LMStudioRequestType<'a> {
    Chat { messages: &'a Value, stream: bool },
    Completion { prompt: Cow<'a, str>, stream: bool },
    Embeddings { input: &'a Value },
}

pub struct TopLevelParams<'a> {
    pub think: Option<&'a Value>,
    pub logprobs: Option<&'a Value>,
    pub top_logprobs: Option<&'a Value>,
    /// Whether the resolved model is reasoning-capable. When `think` is absent
    /// and this is true, the proxy defaults `reasoning` to `"on"`, matching real
    /// Ollama (thinking models reason by default). An explicit `think` always
    /// wins; `think:false` → `"off"`, re-expressed as the model's lowest tier
    /// when it cannot disable thinking (see `reasoning_disable_fallback`).
    pub model_is_thinking: bool,
    /// Tier substituted for a normalized `reasoning:"off"` when the resolved
    /// model cannot disable thinking
    /// (`NativeReasoningCapability::reasoning_disable_fallback`). `None` keeps
    /// the client's value verbatim — models with an off option, an unknown
    /// vocabulary or no reasoning metadata are untouched.
    pub reasoning_disable_fallback: Option<&'a str>,
}

pub fn map_ollama_to_lmstudio_params(
    ollama_options: Option<&Value>,
    structured_format: Option<&Value>,
) -> serde_json::Map<String, Value> {
    let mut params = serde_json::Map::new();

    map_direct_params(ollama_options, &mut params);
    map_token_limits(ollama_options, &mut params);
    map_format_params(ollama_options, structured_format, &mut params);

    if let Some(options) = ollama_options {
        log_unsupported_options(options);
    }

    params
}

/// Normalise an Ollama `think` value to LM Studio's `reasoning` setting.
///
/// LM Studio accepts: off | low | medium | high | on — no `max`. Ollama/OpenAI
/// use `"none"` for disabled (→ `"off"`) and `"max"` for the highest thinking
/// level, which we clamp to LM Studio's top effort tier `"high"` (a bare `"max"`
/// is rejected by the backend with an unsupported-reasoning-setting error).
/// Bools collapse to `"on"`/`"off"`; any other string passes through; unexpected
/// types are forwarded as-is.
pub fn normalize_reasoning(think_val: &Value) -> Value {
    match think_val {
        Value::Bool(true) => json!("on"),
        Value::Bool(false) => json!("off"),
        Value::String(s) if s.eq_ignore_ascii_case("none") => json!("off"),
        Value::String(s) if s.eq_ignore_ascii_case("max") => json!("high"),
        Value::String(s) => json!(s),
        other => {
            log::debug!("think: unexpected value type {:?}, forwarding as-is", other);
            other.clone()
        }
    }
}

fn apply_top_level_params(
    top: &TopLevelParams<'_>,
    request_obj: &mut serde_json::Map<String, Value>,
) {
    if let Some(think_val) = top.think {
        let mut reasoning = normalize_reasoning(think_val);
        // A model that cannot disable thinking has no meaning for "off": send
        // its lowest supported tier instead
        // (api-docs/ollama/capabilities/thinking.md). Only a normalized "off"
        // substitutes — true/"max" and arbitrary model-defined strings pass
        // through untouched.
        if reasoning.as_str() == Some("off")
            && let Some(tier) = top.reasoning_disable_fallback
        {
            reasoning = json!(tier);
        }
        request_obj.insert("reasoning".to_string(), reasoning);
    } else if top.model_is_thinking {
        // No explicit `think`, but the model reasons by default → enable it,
        // mirroring real Ollama. Explicit `think` (handled above) always wins.
        request_obj.insert("reasoning".to_string(), json!("on"));
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
            forward_embeddings_only_params(ollama_options, &mut body);
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
    // Listed in LM Studio's chat-completions doc
    // (api-docs/lmstudio/1_developer/3_openai-compat/chat-completions.md).
    // LM Studio v0 chat accepts `min_p` (verified live), so it is forwarded as a
    // direct sampling key alongside `top_k`.
    const DIRECT_MAPPINGS: &[&str] = &[
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
    ];

    let Some(options) = ollama_options else {
        return;
    };

    for param in DIRECT_MAPPINGS {
        if let Some(value) = options.get(param) {
            params.insert((*param).to_string(), value.clone());
        }
    }

    if let Some(logit_bias) = options.get("logit_bias") {
        params.insert("logit_bias".to_string(), logit_bias.clone());
    }
}

// Ollama spec (api-docs/ollama/api/embed.md) defines `truncate` and
// `dimensions` only for /api/embed. They have no meaning on chat-completions
// and must not pollute the upstream body there.
fn forward_embeddings_only_params(
    ollama_options: Option<&Value>,
    body: &mut serde_json::Map<String, Value>,
) {
    const EMBEDDINGS_ONLY: &[&str] = &["truncate", "dimensions"];

    if let Some(options) = ollama_options {
        for key in EMBEDDINGS_ONLY {
            if let Some(value) = options.get(key) {
                body.insert((*key).to_string(), value.clone());
            }
        }
    }

    // Ollama's documented default is `truncate: true` (api-docs/ollama/api/
    // embed.md); fill the gap only when the caller omitted it so an explicit
    // value (even `false`) always wins.
    body.entry("truncate".to_string())
        .or_insert(serde_json::json!(true));
}

fn map_token_limits(ollama_options: Option<&Value>, params: &mut serde_json::Map<String, Value>) {
    let Some(options) = ollama_options else {
        return;
    };

    if let Some(max_tokens) = options
        .get("max_tokens")
        .or_else(|| options.get("num_predict"))
    {
        // Ollama's num_predict -1 (infinite) / -2 (fill context) are "no limit"
        // sentinels with no LM Studio equivalent; the backend 400s on a negative
        // max_tokens ("must be greater than 0"), so omit the field entirely
        // rather than forward a value that gets rejected.
        let is_no_limit = json_as_i64(max_tokens).is_some_and(|n| n < 0);
        if !is_no_limit {
            params.insert("max_tokens".to_string(), max_tokens.clone());
        }
    }

    // `num_ctx` is NOT emitted here: LM Studio's chat body ignores
    // `context_length` (it is a load-time parameter). It is honored out-of-band
    // by `ensure_context_length`, which reloads the model at the requested size.
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
    // Speculative-decoding draft-token cap. LM Studio configures speculative
    // decoding at model-load time (draft model selection), with no per-request
    // knob in its REST/native API — so this has no upstream equivalent and is
    // warn-dropped like Ollama with no draft model loaded (a no-op there too).
    "draft_num_predict",
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

/// LM Studio's `response_format` accepts only `json_schema` or `text`. Both the
/// Ollama `format:"json"` shim and an OpenAI `response_format:{"type":"json_object"}`
/// map to this permissive object-typed json_schema, so any JSON object validates.
///
/// `name` is a required field of LM Studio's json_schema envelope; Ollama/OpenAI
/// supply none, so it is a synthetic label LM Studio ignores (verified live).
/// Dropping it risks a 400.
pub(crate) fn permissive_json_object_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "json",
            "schema": { "type": "object" }
        }
    })
}

/// Rewrite an OpenAI `response_format:{"type":"json_object"}` in place into LM
/// Studio's permissive json_schema envelope. LM Studio's `/v1` chat/completions
/// rejects a bare `json_object` with a 400. No-op for any other `response_format`
/// shape (an explicit `json_schema` or `text` passes through untouched).
pub(crate) fn rewrite_json_object_format(body: &mut Value) {
    let is_json_object = body
        .get("response_format")
        .and_then(|rf| rf.get("type"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t.eq_ignore_ascii_case("json_object"));
    if is_json_object && let Some(obj) = body.as_object_mut() {
        obj.insert(
            "response_format".to_string(),
            permissive_json_object_response_format(),
        );
    }
}

/// Go's `base64.StdEncoding` as the crate's decoder: Go is non-strict about
/// the final quantum's trailing bits, so a payload like `QR==` decodes where
/// the crate's `STANDARD` engine refuses it.
const GO_BASE64_STANDARD: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        base64::engine::general_purpose::GeneralPurposeConfig::new()
            .with_decode_allow_trailing_bits(true),
    );

/// An image string's base64 payload with the CR/LF wrapping some clients add
/// at line length stripped, as Ollama's image decode tolerates.
fn base64_payload(raw: &str) -> String {
    raw.chars().filter(|c| *c != '\r' && *c != '\n').collect()
}

/// Prepare an Ollama System One body for LM Studio: drop a top-level `stream`
/// key (the endpoint is non-streaming, and the generic funnel would otherwise
/// wrap the JSON answer as SSE) and rewrite bare-base64 `images` entries into
/// the data URLs LM Studio's decision parser accepts. The MIME type comes from
/// the decoded magic bytes, the same wrapping Ollama applies to its decoded
/// image bytes; Ollama additionally re-encodes WebP as PNG before wrapping,
/// while the proxy forwards WebP bytes unchanged.
pub(crate) fn prepare_systemone_request(body: &mut Value) -> Result<(), String> {
    if let Some(object) = body.as_object_mut() {
        object.remove("stream");
    }
    let Some(images) = body.get_mut("images").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for (index, image) in images.iter_mut().enumerate() {
        let Some(raw) = image.as_str() else {
            return Err(format!("images[{index}] must be a base64 string"));
        };
        let cleaned = base64_payload(raw);
        let decoded = GO_BASE64_STANDARD
            .decode(&cleaned)
            .map_err(|_| format!("images[{index}] is not valid base64"))?;
        *image = Value::String(format!(
            "data:{};base64,{cleaned}",
            detect_media_type(&decoded)
        ));
    }
    Ok(())
}

/// System One text-and-schema cap (bytes) for a model with no widened
/// decision family.
pub(crate) const SYSTEMONE_TEXT_LIMIT_BYTES: usize = 64 * 1024;

/// The cap upstream widens to for the `kev` and `d1` renderers.
const SYSTEMONE_KEV_D1_TEXT_LIMIT_BYTES: usize = 1024 * 1024;

/// Text-and-schema cap for a System One request: 64 KiB, widened to 1 MiB when
/// the model's LM Studio `decision_type` is exactly `kev` or `d1` — the
/// renderer check `server/routes.go` SystemOneHandler applies. Images never
/// count against this limit.
pub(crate) fn systemone_text_limit(decision_type: Option<&str>) -> usize {
    match decision_type {
        Some("kev" | "d1") => SYSTEMONE_KEV_D1_TEXT_LIMIT_BYTES,
        _ => SYSTEMONE_TEXT_LIMIT_BYTES,
    }
}

/// Upstream's `decision.Request` (`api.SystemOneRequest`, `api/types.go`): the
/// fields `server/routes.go` SystemOneHandler re-encodes to measure a request's
/// text and schema. `images` decodes as Go's `[]api.ImageData` but never
/// serializes, because upstream clears it before measuring. Decoding follows
/// Go's `encoding/json`: struct keys match case-insensitively and a repeated
/// key overwrites (`decode.go` object/foldName), an absent or `null` string is
/// `""`, `state` and `questions` always marshal (`null` when absent),
/// `runner`/`videos`/`keep_alive` carry `omitempty`, unknown keys drop, and
/// `json.RawMessage` fields keep the client's exact token spellings.
///
/// The counted encoding matches Go's legacy `encoding/json` v1 — the
/// implementation in the toolchain Ollama pins (go1.26.10) — on the measured
/// corpus. The outer
/// encode applies `SetEscapeHTML(false)` but still escapes U+2028/U+2029 in the
/// typed `model`/`runner` strings, the `questions` subtree re-encodes through
/// `internal/orderedmap`'s plain `json.Marshal` with HTML escaping on, and raw
/// `state`/`videos` values stay byte-exact apart from compacted whitespace.
/// Residual divergences: a Go 1.27+ json-v2 build also escapes literal
/// U+2028/U+2029 inside raw values where v1 keeps them; a body this mirror
/// cannot decode (invalid UTF-8, or a lone UTF-16 surrogate escape in a typed
/// string) answers the proxy's own 400 where Go decodes it (substituting U+FFFD
/// in a typed string), and a body this mirror decodes but serde's `Value` parser
/// refuses (a surrogate escape kept verbatim inside a raw value, nesting past
/// serde's recursion limit) meets the passthrough's JSON re-parse, which answers
/// 400 `invalid JSON body: …` where Go caps or serves it. The runner refusal quotes with
/// Rust's `{:?}`, which is Go's `%q` for printable text and differs only in
/// non-printable and non-ASCII quoting; the status stays 400 either way.
#[derive(Debug, Default)]
pub(crate) struct SystemOneMirror {
    model: String,
    runner: String,
    state: GoRaw,
    videos: Option<Vec<Box<RawValue>>>,
    questions: Option<BTreeMap<String, SystemOneQuestionMirror>>,
    keep_alive: Option<GoDuration>,
}

impl SystemOneMirror {
    /// Decode a body the way Go's `json.Unmarshal` does: an object decodes
    /// field by field, JSON `null` is a no-op that leaves the zero request, and
    /// every other top-level shape is a decode error. Go's scanner caps nesting
    /// at 10000 containers, so a deeper body is refused before parsing.
    pub(crate) fn parse(body: &[u8]) -> Result<Self, String> {
        if go_nesting_depth_exceeded(body) {
            return Err("json: exceeded max depth".to_string());
        }
        Ok(serde_json::from_slice::<Option<Self>>(body)
            .map_err(|error| error.to_string())?
            .unwrap_or_default())
    }

    /// The size upstream compares with its cap: `len(bytes.TrimSpace(encoded))`
    /// over the compact re-encoding of this request.
    pub(crate) fn text_size(&self) -> usize {
        let encoded = serde_json::to_string(self).expect("a System One request mirror serializes");
        go_compact(&encoded).len()
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn runner(&self) -> &str {
        &self.runner
    }
}

impl Serialize for SystemOneMirror {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut request = serializer.serialize_struct("SystemOneMirror", 6)?;
        request.serialize_field("model", &GoString(&self.model))?;
        if !self.runner.is_empty() {
            request.serialize_field("runner", &GoString(&self.runner))?;
        }
        request.serialize_field("state", &self.state)?;
        if let Some(videos) = &self.videos
            && !videos.is_empty()
        {
            request.serialize_field("videos", videos)?;
        }
        request.serialize_field("questions", &self.questions.as_ref().map(GoQuestions))?;
        if let Some(keep_alive) = &self.keep_alive {
            request.serialize_field("keep_alive", keep_alive)?;
        }
        request.end()
    }
}

/// A Go `string` field as the outer encode writes it: Go's v1 encoder runs with
/// `SetEscapeHTML(false)` there, so only U+2028/U+2029 differ from serde's
/// escaping — Go escapes them unconditionally, six bytes against three.
struct GoString<'a>(&'a str);

impl Serialize for GoString<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let quoted = serde_json::to_string(self.0).map_err(S::Error::custom)?;
        RawValue::from_string(escape_separators(&quoted))
            .map_err(S::Error::custom)?
            .serialize(serializer)
    }
}

/// The `questions` subtree as upstream writes it: `internal/orderedmap.Map`'s
/// `MarshalJSON` is a plain `json.Marshal`, so its own encode keeps HTML
/// escaping on — `<`, `>` and `&` become six-byte unicode escapes across names,
/// `type` and the raw `instructions`/`criteria` values, independent of the
/// outer `SetEscapeHTML(false)`.
struct GoQuestions<'a>(&'a BTreeMap<String, SystemOneQuestionMirror>);

impl Serialize for GoQuestions<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = serde_json::to_string(self.0).map_err(S::Error::custom)?;
        let escaped = encoded
            .replace('<', "\\u003c")
            .replace('>', "\\u003e")
            .replace('&', "\\u0026");
        RawValue::from_string(escape_separators(&escaped))
            .map_err(S::Error::custom)?
            .serialize(serializer)
    }
}

/// Rewrite literal U+2028/U+2029 as Go's six-byte unicode escapes.
fn escape_separators(encoded: &str) -> String {
    encoded
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

impl<'de> Deserialize<'de> for SystemOneMirror {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MirrorVisitor;

        impl<'de> Visitor<'de> for MirrorVisitor {
            type Value = SystemOneMirror;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a System One request object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut request = SystemOneMirror::default();
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    // Go tries the exact field name, then its case-folded form
                    // (`encoding/json` object/foldName); the names are ASCII,
                    // so ASCII case folding is the whole rule.
                    if go_key_matches(key.as_ref(), "model") {
                        request.model = map.next_value::<Option<String>>()?.unwrap_or_default();
                    } else if go_key_matches(key.as_ref(), "runner") {
                        request.runner = map.next_value::<Option<String>>()?.unwrap_or_default();
                    } else if go_key_matches(key.as_ref(), "state") {
                        request.state = map.next_value()?;
                    } else if go_key_matches(key.as_ref(), "videos") {
                        request.videos = map.next_value()?;
                    } else if go_key_matches(key.as_ref(), "questions") {
                        // Go's orderedmap.Map.UnmarshalJSON allocates a fresh
                        // inner map on every occurrence, so a repeated
                        // top-level `questions` field keeps only the last map.
                        match map
                            .next_value::<Option<BTreeMap<String, SystemOneQuestionMirror>>>()?
                        {
                            Some(entries) => request.questions = Some(entries),
                            None => request.questions = None,
                        }
                    } else if go_key_matches(key.as_ref(), "keep_alive") {
                        request.keep_alive = map.next_value()?;
                    } else if go_key_matches(key.as_ref(), "images") {
                        validate_systemone_images(map.next_value()?)?;
                    } else {
                        map.next_value::<de::IgnoredAny>()?;
                    }
                }
                Ok(request)
            }
        }

        deserializer.deserialize_map(MirrorVisitor)
    }
}

/// One `SystemOneQuestion` (`api/types.go`) as Go decodes it: a JSON `null`
/// question is the zero question, keys match case-insensitively and a repeated
/// key overwrites, unknown keys drop, `type` is always emitted, `instructions`
/// always marshals (`null` when absent), and `criteria` carries `omitempty` —
/// absent drops it while a present `null` emits.
#[derive(Debug, Default, Serialize)]
struct SystemOneQuestionMirror {
    #[serde(rename = "type")]
    kind: String,
    instructions: GoRaw,
    #[serde(skip_serializing_if = "GoRaw::is_absent")]
    criteria: GoRaw,
}

impl<'de> Deserialize<'de> for SystemOneQuestionMirror {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct QuestionVisitor;

        impl<'de> Visitor<'de> for QuestionVisitor {
            type Value = SystemOneQuestionMirror;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a System One question object or null")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SystemOneQuestionMirror::default())
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SystemOneQuestionMirror::default())
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut question = SystemOneQuestionMirror::default();
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    if go_key_matches(key.as_ref(), "type") {
                        question.kind = map.next_value::<Option<String>>()?.unwrap_or_default();
                    } else if go_key_matches(key.as_ref(), "instructions") {
                        question.instructions = map.next_value()?;
                    } else if go_key_matches(key.as_ref(), "criteria") {
                        question.criteria = map.next_value()?;
                    } else {
                        map.next_value::<de::IgnoredAny>()?;
                    }
                }
                Ok(question)
            }
        }

        deserializer.deserialize_any(QuestionVisitor)
    }
}

/// Go's `encoding/json` field match (`decode.go` object/foldName): ASCII case
/// folding, plus the two non-ASCII runes whose Unicode simple fold set holds an
/// ASCII letter — KELVIN SIGN with `k`, LONG S with `s`. With ASCII field names
/// no other rune can fold onto one.
fn go_key_matches(key: &str, field: &str) -> bool {
    key.chars()
        .map(|character| match character {
            '\u{212A}' => 'k',
            '\u{017F}' => 's',
            character => character.to_ascii_lowercase(),
        })
        .eq(field.chars())
}

/// One Go `json.RawMessage` field: absent, a present JSON `null`, or a raw JSON
/// token. The three are distinct because `omitempty` drops an absent field but
/// not a present `null`, which marshals as `null`.
#[derive(Debug, Default)]
enum GoRaw {
    #[default]
    Absent,
    Null,
    Value(Box<RawValue>),
}

impl GoRaw {
    fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }
}

impl<'de> Deserialize<'de> for GoRaw {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        Ok(if raw.get() == "null" {
            Self::Null
        } else {
            Self::Value(raw)
        })
    }
}

impl Serialize for GoRaw {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Value(raw) => raw.serialize(serializer),
            Self::Absent | Self::Null => serializer.serialize_none(),
        }
    }
}

/// Decode a `SystemOneRequest.images` field the way Go decodes
/// `[]api.ImageData`: absent or `null` is empty, each element is a base64
/// string (or `null`), and any other shape fails the decode. The bytes are
/// dropped — upstream clears images before measuring, so they never count.
fn validate_systemone_images<E>(images: Option<Vec<Option<String>>>) -> Result<(), E>
where
    E: de::Error,
{
    let Some(images) = images else {
        return Ok(());
    };
    for (index, image) in images.iter().enumerate() {
        let Some(raw) = image else {
            continue;
        };
        GO_BASE64_STANDARD
            .decode(base64_payload(raw))
            .map_err(|_| E::custom(format!("images[{index}] is not valid base64")))?;
    }
    Ok(())
}

/// Go's `api.Duration` (`api/types.go`): JSON numbers are seconds, strings
/// parse with `time.ParseDuration`, and a negative duration clamps to
/// `math.MaxInt64`. The encoder writes `"<time.Duration.String()>"`, so `300`
/// becomes `"5m0s"`; the one negative that survives decoding — a float64 to
/// int64 conversion out of range — marshals as the number `-1`.
#[derive(Clone, Copy, Debug)]
struct GoDuration {
    nanos: i64,
}

/// One second in nanoseconds.
const NANOS_PER_SECOND: f64 = 1e9;

/// `math.MaxInt64 + 1`; at or above it the float64-to-int64 conversion is out
/// of range, as `1 << 63` is for an integer conversion.
const GO_DURATION_OVERFLOW: f64 = (1u64 << 63) as f64;

impl GoDuration {
    /// `time.Duration(t * float64(time.Second))` with `api.Duration`'s negative
    /// clamp. Go leaves the out-of-range conversion implementation-dependent;
    /// on the amd64 reference it yields `math.MinInt64`, which marshals as `-1`.
    fn from_seconds(seconds: f64) -> Self {
        if seconds < 0.0 {
            return Self { nanos: i64::MAX };
        }
        let nanos = seconds * NANOS_PER_SECOND;
        if nanos >= GO_DURATION_OVERFLOW {
            return Self { nanos: i64::MIN };
        }
        Self {
            nanos: nanos as i64,
        }
    }

    /// `time.ParseDuration`, then `api.Duration`'s negative clamp.
    fn parse(text: &str) -> Result<Self, String> {
        let nanos = parse_go_duration(text)?;
        Ok(Self {
            nanos: if nanos < 0 { i64::MAX } else { nanos },
        })
    }

    /// `time.Duration.String()` (`time/time.go`): `"72h3m0.5s"`, a smaller unit
    /// below one second (`"500ms"`, `"1.5µs"`, `"1ns"`), and `"0s"` for zero.
    fn format(self) -> String {
        const SECOND: u64 = 1_000_000_000;
        const MICROSECOND: u64 = 1_000;
        const MILLISECOND: u64 = 1_000_000;

        let mut buffer = [0u8; 32];
        let mut write = buffer.len();
        let mut value = self.nanos as u64;
        let negative = self.nanos < 0;
        if negative {
            value = value.wrapping_neg();
        }
        if value < SECOND {
            write -= 1;
            buffer[write] = b's';
            write -= 1;
            let precision = match value {
                0 => {
                    buffer[write] = b'0';
                    return duration_text(&buffer[write..]);
                }
                value if value < MICROSECOND => {
                    buffer[write] = b'n';
                    0
                }
                value if value < MILLISECOND => {
                    // U+00B5 'µ' takes two bytes.
                    write -= 1;
                    buffer[write] = 0xC2;
                    buffer[write + 1] = 0xB5;
                    3
                }
                _ => {
                    buffer[write] = b'm';
                    6
                }
            };
            (write, value) = format_fraction(&mut buffer, write, value, precision);
            write = format_integer(&mut buffer, write, value);
        } else {
            write -= 1;
            buffer[write] = b's';
            (write, value) = format_fraction(&mut buffer, write, value, 9);
            write = format_integer(&mut buffer, write, value % 60);
            value /= 60;
            if value > 0 {
                write -= 1;
                buffer[write] = b'm';
                write = format_integer(&mut buffer, write, value % 60);
                value /= 60;
                if value > 0 {
                    write -= 1;
                    buffer[write] = b'h';
                    write = format_integer(&mut buffer, write, value);
                }
            }
        }
        if negative {
            write -= 1;
            buffer[write] = b'-';
        }
        duration_text(&buffer[write..])
    }
}

impl<'de> Deserialize<'de> for GoDuration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct GoDurationVisitor;

        impl<'de> Visitor<'de> for GoDurationVisitor {
            type Value = GoDuration;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a number of seconds or a Go duration string")
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(GoDuration::from_seconds(v as f64))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(GoDuration::from_seconds(v as f64))
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(GoDuration::from_seconds(v))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                GoDuration::parse(v).map_err(E::custom)
            }

            fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Err(E::custom("Unsupported type: 'bool'"))
            }

            fn visit_seq<A>(self, _: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                Err(<A::Error as de::Error>::custom(
                    "Unsupported type: '[]interface {}'",
                ))
            }

            fn visit_map<A>(self, _: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                Err(<A::Error as de::Error>::custom(
                    "Unsupported type: 'map[string]interface {}'",
                ))
            }
        }

        deserializer.deserialize_any(GoDurationVisitor)
    }
}

impl Serialize for GoDuration {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.nanos < 0 {
            return serializer.serialize_i64(-1);
        }
        serializer.serialize_str(&self.format())
    }
}

/// Byte size upstream's System One handler measures for a body — `None` when
/// Go's decode would reject it, so the caller answers upstream's 400 instead of
/// measuring it.
#[cfg(test)]
pub(crate) fn systemone_text_size(body: &[u8]) -> Option<usize> {
    SystemOneMirror::parse(body)
        .ok()
        .map(|request| request.text_size())
}

/// Upstream's `normalizeRunner` (`server/routes.go`): trim and lowercase, then
/// accept the empty runner and Ollama's runner spellings. An unknown runner
/// answers upstream's error, quoting the original untrimmed value.
pub(crate) fn systemone_runner_error(runner: &str) -> Option<String> {
    match runner.trim().to_lowercase().as_str() {
        "" | "mlx" | "mlxrunner" | "ggml" | "llamacpp" | "llama.cpp" | "llama-cpp"
        | "llama_cpp" => None,
        _ => Some(format!("unknown runner {runner:?}")),
    }
}

/// Upstream's cloud-ref refusal (`internal/modelref.ParseRef` plus the
/// `ref.Source == modelSourceCloud` check in `server/routes.go`
/// SystemOneHandler): the decision has to run locally. A ref is cloud when the
/// suffix after its last ':' is `cloud` or ends in `-cloud` without a '/'.
pub(crate) fn systemone_model_ref_error(model: &str) -> Option<&'static str> {
    let (_, suffix) = model.trim().rsplit_once(':')?;
    let suffix = suffix.trim().to_lowercase();
    let cloud = suffix == "cloud" || (suffix.ends_with("-cloud") && !suffix.contains('/'));
    cloud.then_some("System One requires a local decision model")
}

/// `time.ParseDuration` (`time/format.go`) over nanoseconds, with Go's error
/// text. `Ok` may be `i64::MIN` for a negative duration of exactly 2^63 ns;
/// the caller applies `api.Duration`'s negative clamp.
fn parse_go_duration(text: &str) -> Result<i64, String> {
    const NANOSECOND: u64 = 1;
    const MICROSECOND: u64 = 1_000;
    const MILLISECOND: u64 = 1_000_000;
    const SECOND: u64 = 1_000_000_000;
    const MINUTE: u64 = 60 * SECOND;
    const HOUR: u64 = 60 * MINUTE;
    const MAX_INT64: u64 = 1 << 63;

    let original = text;
    let mut rest = text.as_bytes();
    let mut total: u64 = 0;
    let mut negative = false;

    if let Some((&sign, tail)) = rest.split_first()
        && (sign == b'-' || sign == b'+')
    {
        negative = sign == b'-';
        rest = tail;
    }
    if rest == b"0" {
        return Ok(0);
    }
    if rest.is_empty() {
        return Err(invalid_duration(original));
    }
    while !rest.is_empty() {
        if !(rest[0] == b'.' || rest[0].is_ascii_digit()) {
            return Err(invalid_duration(original));
        }
        let before = rest.len();
        let (mut value, tail) = leading_int(rest).ok_or_else(|| invalid_duration(original))?;
        rest = tail;
        let had_integer = before != rest.len();

        let mut fraction: u64 = 0;
        let mut scale: f64 = 1.0;
        let had_fraction = if rest.first() == Some(&b'.') {
            rest = &rest[1..];
            let before = rest.len();
            (fraction, scale, rest) = leading_fraction(rest);
            before != rest.len()
        } else {
            false
        };
        if !had_integer && !had_fraction {
            return Err(invalid_duration(original));
        }

        let mut unit_end = 0;
        while unit_end < rest.len() && rest[unit_end] != b'.' && !rest[unit_end].is_ascii_digit() {
            unit_end += 1;
        }
        if unit_end == 0 {
            return Err(format!("time: missing unit in duration {original:?}"));
        }
        let unit_name = &rest[..unit_end];
        rest = &rest[unit_end..];
        let unit = match unit_name {
            b"ns" => NANOSECOND,
            b"us" => MICROSECOND,
            b"ms" => MILLISECOND,
            b"s" => SECOND,
            b"m" => MINUTE,
            b"h" => HOUR,
            _ if unit_name == "µs".as_bytes() || unit_name == "μs".as_bytes() => MICROSECOND,
            _ => {
                return Err(format!(
                    "time: unknown unit {:?} in duration {original:?}",
                    String::from_utf8_lossy(unit_name)
                ));
            }
        };
        if value > MAX_INT64 / unit {
            return Err(invalid_duration(original));
        }
        value *= unit;
        if fraction > 0 {
            // Go keeps nanosecond accuracy through float64 here; the fraction is
            // below one, so the product stays far inside u64.
            value += (fraction as f64 * (unit as f64 / scale)) as u64;
            if value > MAX_INT64 {
                return Err(invalid_duration(original));
            }
        }
        total = total.wrapping_add(value);
        if total > MAX_INT64 {
            return Err(invalid_duration(original));
        }
    }
    if negative {
        // Go's `-Duration(d)` wraps when d is exactly 2^63.
        let duration = i64::try_from(total).unwrap_or(i64::MIN);
        return Ok(duration.wrapping_neg());
    }
    if total > MAX_INT64 - 1 {
        return Err(invalid_duration(original));
    }
    Ok(i64::try_from(total).expect("total is at most 2^63 - 1 after the overflow check"))
}

/// Go's `leadingInt`: the leading `[0-9]*`, or `None` on its overflow.
fn leading_int(bytes: &[u8]) -> Option<(u64, &[u8])> {
    let mut value: u64 = 0;
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        if !byte.is_ascii_digit() {
            break;
        }
        if value > (1u64 << 63) / 10 {
            return None;
        }
        value = value * 10 + u64::from(byte - b'0');
        if value > 1u64 << 63 {
            return None;
        }
        index += 1;
    }
    Some((value, &bytes[index..]))
}

/// Go's `leadingFraction`: the leading `[0-9]*` as a value and its power-of-ten
/// scale. Digits past the first overflow only advance the scan.
fn leading_fraction(bytes: &[u8]) -> (u64, f64, &[u8]) {
    let mut value: u64 = 0;
    let mut scale: f64 = 1.0;
    let mut overflow = false;
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        if !byte.is_ascii_digit() {
            break;
        }
        if !overflow {
            if value > ((1u64 << 63) - 1) / 10 {
                overflow = true;
            } else {
                let next = value * 10 + u64::from(byte - b'0');
                if next > 1u64 << 63 {
                    overflow = true;
                } else {
                    value = next;
                    scale *= 10.0;
                }
            }
        }
        index += 1;
    }
    (value, scale, &bytes[index..])
}

fn invalid_duration(original: &str) -> String {
    format!("time: invalid duration {original:?}")
}

/// Go's `fmtFrac`: the fraction digits of `value / 10**precision` written
/// before `write`, trailing zeros omitted along with the decimal point.
fn format_fraction(
    buffer: &mut [u8],
    mut write: usize,
    mut value: u64,
    precision: u32,
) -> (usize, u64) {
    let mut print = false;
    for _ in 0..precision {
        let digit = value % 10;
        print = print || digit != 0;
        if print {
            write -= 1;
            buffer[write] = b'0' + u8::try_from(digit).expect("a decimal digit fits in u8");
        }
        value /= 10;
    }
    if print {
        write -= 1;
        buffer[write] = b'.';
    }
    (write, value)
}

/// Go's `fmtInt`: `value` in decimal written before `write`.
fn format_integer(buffer: &mut [u8], mut write: usize, mut value: u64) -> usize {
    if value == 0 {
        write -= 1;
        buffer[write] = b'0';
    } else {
        while value > 0 {
            write -= 1;
            buffer[write] = b'0' + u8::try_from(value % 10).expect("a decimal digit fits in u8");
            value /= 10;
        }
    }
    write
}

fn duration_text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec())
        .expect("a duration spelling is ASCII plus the two-byte micro sign")
}

/// Go's `encoding/json` scanner nesting cap (`scanner.go` `maxNestingDepth`).
const GO_JSON_MAX_NESTING_DEPTH: usize = 10_000;

/// Whether the body nests containers deeper than Go's scanner allows. Strings
/// are skipped, so a bracket inside one does not count.
fn go_nesting_depth_exceeded(body: &[u8]) -> bool {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in body {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > GO_JSON_MAX_NESTING_DEPTH {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

/// Go's `encoding/json` compact step (`appendCompact` in `indent.go`) with
/// `escape` off, as the encoder applies it to a `json.RawMessage`: drop
/// insignificant whitespace outside strings and copy every byte inside one,
/// escape sequences included.
fn go_compact(encoded: &str) -> String {
    let mut compacted = Vec::with_capacity(encoded.len());
    let mut in_string = false;
    let mut escaped = false;
    for &byte in encoded.as_bytes() {
        if in_string {
            compacted.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
            compacted.push(byte);
        } else if !matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
            compacted.push(byte);
        }
    }
    String::from_utf8(compacted).expect("compacting valid UTF-8 JSON keeps it valid UTF-8")
}

/// Media type from an image's magic bytes for the types Ollama documents (PNG,
/// JPEG, WebP) plus the GIF/BMP/WAV cases Go's `http.DetectContentType`
/// assigns; every other payload (ICO, TIFF, text, …) gets
/// `application/octet-stream`.
fn detect_media_type(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        "image/png"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        "image/gif"
    } else if bytes.starts_with(b"RIFF") {
        match bytes.get(8..12) {
            Some(b"WEBP") => "image/webp",
            Some(b"WAVE") => "audio/wave",
            _ => "application/octet-stream",
        }
    } else if bytes.starts_with(b"BM") {
        "image/bmp"
    } else {
        "application/octet-stream"
    }
}

fn convert_structured_format(format_value: &Value) -> Option<Value> {
    match format_value {
        Value::String(mode) if mode.eq_ignore_ascii_case("json") => {
            Some(permissive_json_object_response_format())
        }
        Value::String(mode) if mode.eq_ignore_ascii_case("text") => Some(json!({ "type": "text" })),
        Value::Object(_) => {
            let mut json_schema = serde_json::Map::new();
            // `name` is a required field of LM Studio's json_schema envelope;
            // Ollama supplies none, so this is a synthetic label LM Studio
            // ignores (verified live). Dropping it risks a 400.
            json_schema.insert("name".to_string(), json!("ollama_format"));
            // `strict:true` forces every declared property into `required`; on a
            // schema with an optional field that 400s. Only send strict when the
            // schema already requires all its properties (recursively).
            if all_properties_required(format_value) {
                json_schema.insert("strict".to_string(), json!(true));
            }
            json_schema.insert("schema".to_string(), format_value.clone());
            Some(json!({ "type": "json_schema", "json_schema": json_schema }))
        }
        _ => None,
    }
}

/// JSON-Schema keywords whose value is itself a schema (or holds one) that
/// this walker does not open: `$ref`/`$dynamicRef` point elsewhere and are
/// never resolved here; the rest (`patternProperties`, `propertyNames`,
/// `contains`, `not`, `dependentSchemas`, `if`/`then`/`else`) can each carry a
/// nested `properties`/`required` pair this function never sees. `strict` is
/// an optimization: declining to prove safety only skips an enforcement, it
/// never causes a bad request, so any node carrying one of these keywords
/// counts as unproven rather than safe.
///
/// `contains`/`not` stay gated rather than walked even though their value is
/// a schema: `contains` is existential (at least one array item must match)
/// and `not` describes a forbidden, not an alternate valid, shape, so
/// proving either nested schema all-required says nothing sound about the
/// outer schema, unlike `prefixItems`/`items` where every position applies
/// simultaneously.
///
/// `additionalProperties`/`unevaluatedProperties`/`unevaluatedItems` are not
/// here: unlike the keywords above, their value can be a plain boolean, which
/// declares no properties either way and would make this list fail closed on
/// the common fully-closed case (pydantic `extra="forbid"` ->
/// `additionalProperties: false` with every field required). When their
/// value IS a schema (pydantic's `dict[str, Model]` ->
/// `additionalProperties: {"$ref": ...}`), it is a nested object schema by
/// the same standard as any `properties` value or `items`, so it is walked
/// below instead of gated on presence.
///
/// `prefixItems` (2020-12's tuple-validation keyword, replacing draft-07's
/// array-form `items`) is also not here for the same reason as `items`: its
/// value is always an array of per-position schemas that all apply
/// simultaneously, so it is walked alongside `anyOf`/`oneOf`/`allOf` below
/// rather than gated on presence.
const UNPROVEN_SCHEMA_KEYWORDS: &[&str] = &[
    "$ref",
    "$dynamicRef",
    "patternProperties",
    "propertyNames",
    "contains",
    "not",
    "dependentSchemas",
    "if",
    "then",
    "else",
];

/// Keywords covering keys not named in `properties`, whose value can be
/// either a boolean (proves nothing, ignored) or a schema (a nested object
/// schema that must be proven). Recursing unconditionally handles both: the
/// non-object base case of `all_properties_required` already treats a
/// boolean value as trivially safe, exactly the "ignore" a boolean warrants.
const SCHEMA_VALUED_EXTRA_KEYWORDS: &[&str] = &[
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// Whether `strict:true` is safe for this json-schema. LM Studio's strict mode
/// requires every declared property to also appear in its object's `required`
/// list, recursively through every subschema position the schema can express
/// one in: `properties` values, `items` (both the single-schema and
/// draft-07 tuple-array forms), `prefixItems`, each `anyOf`/`oneOf`/`allOf`
/// branch, and the schema-valued form of `SCHEMA_VALUED_EXTRA_KEYWORDS`. A
/// position this function cannot walk (see `UNPROVEN_SCHEMA_KEYWORDS`) makes
/// the whole node unproven, not safe by default, so a schema expressing
/// optionality through some other keyword can't slip through as all-required.
fn all_properties_required(schema: &Value) -> bool {
    let Some(map) = schema.as_object() else {
        return true;
    };

    if UNPROVEN_SCHEMA_KEYWORDS
        .iter()
        .any(|&k| map.contains_key(k))
    {
        return false;
    }

    if let Some(props) = map.get("properties").and_then(|p| p.as_object()) {
        let required: std::collections::HashSet<&str> = map
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if !props.keys().all(|k| required.contains(k.as_str())) {
            return false;
        }
        if !props.values().all(all_properties_required) {
            return false;
        }
    }

    if let Some(items) = map.get("items") {
        let items_safe = match items.as_array() {
            Some(tuple_schemas) => tuple_schemas.iter().all(all_properties_required),
            None => all_properties_required(items),
        };
        if !items_safe {
            return false;
        }
    }

    for keyword in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(branches) = map.get(keyword).and_then(|v| v.as_array())
            && !branches.iter().all(all_properties_required)
        {
            return false;
        }
    }

    for keyword in SCHEMA_VALUED_EXTRA_KEYWORDS {
        if let Some(value) = map.get(*keyword)
            && !all_properties_required(value)
        {
            return false;
        }
    }

    true
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_request.rs"]
mod tests;
