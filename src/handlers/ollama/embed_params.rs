//! Helpers for /api/embed and /api/embeddings request normalization.

use serde_json::{Value, json};

/// Lift Ollama's top-level `/api/embed` advanced parameters (`truncate`, `dimensions`)
/// into the `options` map so the shared option-mapper picks them up.
///
/// Per Ollama spec (api_docs/ollama.md §"Generate Embeddings"), `truncate` and
/// `dimensions` sit at the top level of the request body, peers of `model` and
/// `input`. Values inside an existing `options` object take precedence.
pub fn lift_embed_top_level_params(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    let truncate = obj.remove("truncate");
    let dimensions = obj.remove("dimensions");
    if truncate.is_none() && dimensions.is_none() {
        return;
    }

    let options_entry = obj.entry("options").or_insert_with(|| json!({}));
    let Some(options) = options_entry.as_object_mut() else {
        // `options` is set to a non-object value; restore top-level fields and bail.
        if let Some(t) = truncate {
            obj.insert("truncate".to_string(), t);
        }
        if let Some(d) = dimensions {
            obj.insert("dimensions".to_string(), d);
        }
        return;
    };

    if let Some(t) = truncate {
        options.entry("truncate".to_string()).or_insert(t);
    }
    if let Some(d) = dimensions {
        options.entry("dimensions".to_string()).or_insert(d);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_embed_params.rs"]
mod tests;
