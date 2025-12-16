/// Native LM Studio API endpoints
pub const LM_STUDIO_NATIVE_MODELS: &str = "/api/v1/models";
pub const LM_STUDIO_NATIVE_CHAT: &str = "/v1/chat/completions";
pub const LM_STUDIO_NATIVE_COMPLETIONS: &str = "/v1/completions";
pub const LM_STUDIO_NATIVE_EMBEDDINGS: &str = "/v1/embeddings";
pub const LM_STUDIO_NATIVE_DOWNLOAD: &str = "/api/v1/models/download";
pub const LM_STUDIO_NATIVE_DOWNLOAD_STATUS: &str = "/api/v1/models/download/status";

/// Latest upstream Ollama server version for compatibility endpoints
pub const OLLAMA_SERVER_VERSION: &str = "0.13.0";

/// Timing and performance constants
pub const TOKEN_TO_CHAR_RATIO: f64 = 0.25;
pub const DEFAULT_LOAD_DURATION_NS: u64 = 1_000_000;
pub const TIMING_EVAL_RATIO: u64 = 2;
pub const TIMING_PROMPT_RATIO: u64 = 4;

/// Response headers
pub const CONTENT_TYPE_JSON: &str = "application/json; charset=utf-8";
pub const CONTENT_TYPE_SSE: &str = "text/event-stream";
pub const HEADER_CACHE_CONTROL: &str = "no-cache";
pub const HEADER_CONNECTION: &str = "keep-alive";
pub const HEADER_ACCESS_CONTROL_ALLOW_ORIGIN: &str = "*";
pub const HEADER_ACCESS_CONTROL_ALLOW_METHODS: &str = "GET, POST, PUT, DELETE, OPTIONS";
pub const HEADER_ACCESS_CONTROL_ALLOW_HEADERS: &str = "Content-Type, Authorization";

/// Default parameter values
pub const DEFAULT_TEMPERATURE: f64 = 0.7;
pub const DEFAULT_TOP_P: f64 = 0.9;
pub const DEFAULT_TOP_K: u32 = 40;
pub const DEFAULT_REPEAT_PENALTY: f64 = 1.1;
pub const DEFAULT_KEEP_ALIVE_MINUTES: i64 = 5;

/// Error messages
pub const ERROR_MISSING_MODEL: &str = "Missing 'model' field";
pub const ERROR_MISSING_MESSAGES: &str = "Missing 'messages' field";
pub const ERROR_MISSING_PROMPT: &str = "Missing 'prompt' field";
pub const ERROR_MISSING_INPUT: &str = "Missing 'input' or 'prompt' field";
pub const ERROR_TIMEOUT: &str = "Stream timeout";
pub const ERROR_CANCELLED: &str = "Request cancelled by client";
pub const ERROR_LM_STUDIO_UNAVAILABLE: &str = "LM Studio not available";

/// SSE parsing constants
pub const SSE_DATA_PREFIX: &str = "data: ";
pub const SSE_DONE_MESSAGE: &str = "[DONE]";
pub const SSE_MESSAGE_BOUNDARY: &str = "\n\n";

/// Logging prefixes
pub const LOG_PREFIX_SUCCESS: &str = "✅";
pub const LOG_PREFIX_ERROR: &str = "❌";
pub const LOG_PREFIX_WARNING: &str = "⚠️";
pub const LOG_PREFIX_INFO: &str = "ℹ️";
pub const LOG_PREFIX_CONN: &str = "↔️";

/// Maximum accepted JSON body size (bytes)
pub const MAX_JSON_BODY_SIZE_BYTES: u64 = 16 * 1024 * 1024;
