//! Parse Ollama-style parameter-size strings ("7B", "1.5B", "70B", "500M")
//! into a u64 count of weights.
//!
//! Ollama's `general.parameter_count` field is documented as a raw count
//! (e.g. 8030261248 for an 8.0B model — `api-docs/ollama.md` line 1485).
//! LM Studio's `params_string` is a human-friendly shorthand; this helper
//! converts between the two formats.

/// Parse a string like "7B", "1.5B", "70B", "500M", "0.5B" into a parameter
/// count. Trailing whitespace and case are tolerated. Returns `None` for
/// unrecognized or "unknown" inputs.
pub fn parse_parameter_count(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("unknown") {
        return None;
    }

    let upper = trimmed.to_ascii_uppercase();
    let (number_part, multiplier) = if let Some(rest) = upper.strip_suffix('B') {
        (rest, 1_000_000_000_u64)
    } else if let Some(rest) = upper.strip_suffix('M') {
        (rest, 1_000_000_u64)
    } else if let Some(rest) = upper.strip_suffix('K') {
        (rest, 1_000_u64)
    } else {
        // bare number — treat as count
        (upper.as_str(), 1_u64)
    };

    let number: f64 = number_part.trim().parse().ok()?;
    if !number.is_finite() || number < 0.0 {
        return None;
    }
    Some((number * multiplier as f64).round() as u64)
}

#[cfg(test)]
#[path = "../../tests/unit/model_param_count.rs"]
mod tests;
