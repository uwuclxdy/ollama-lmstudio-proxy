//! Deterministic model-name matching.
//!
//! Given an Ollama-style model name (already cleaned of `:latest` suffixes etc.)
//! and a slice of LM Studio models, pick the single best target. The ordering
//! is fully deterministic — the same input always produces the same output —
//! so model resolution does not depend on LM Studio's response ordering.

#[derive(Debug, Clone)]
pub struct ModelMatchView {
    pub id: String,
    pub arch: String,
    pub model_type: String,
    pub is_loaded: bool,
}

/// Return the best match for `query` among `models`, or `None` if no candidate
/// is plausible. Precedence:
///   1. exact match (case-insensitive)
///   2. substring match — pick loaded over not, then shortest id, then lex order
///   3. scored token-overlap match (≥ 3 points)
pub fn find_best_match<'a>(
    query: &str,
    models: &'a [ModelMatchView],
) -> Option<&'a ModelMatchView> {
    if models.is_empty() {
        return None;
    }
    let lower_query = query.to_lowercase();
    let lowered_ids: Vec<String> = models.iter().map(|m| m.id.to_lowercase()).collect();

    // 1. Exact (case-insensitive) match.
    for (i, lowered) in lowered_ids.iter().enumerate() {
        if *lowered == lower_query {
            return Some(&models[i]);
        }
    }

    // 2. Substring matches — collect all then break ties deterministically.
    let substring_idxs: Vec<usize> = lowered_ids
        .iter()
        .enumerate()
        .filter(|(i, lowered)| {
            lowered.contains(&*lower_query)
                && (lower_query.len() > models[*i].id.len() / 2 || lower_query.len() > 10)
        })
        .map(|(i, _)| i)
        .collect();

    if !substring_idxs.is_empty() {
        // loaded > shortest id length > lex(id)
        let best = substring_idxs
            .iter()
            .min_by(|&&a, &&b| {
                let ma = &models[a];
                let mb = &models[b];
                ma.is_loaded
                    .cmp(&mb.is_loaded)
                    .reverse()
                    .then_with(|| ma.id.len().cmp(&mb.id.len()))
                    .then_with(|| ma.id.cmp(&mb.id))
            })
            .copied()
            .unwrap();
        return Some(&models[best]);
    }

    // 3. Token-overlap scored match, with deterministic tiebreak (lex id).
    let mut scored: Vec<(usize, usize)> = models
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let score = calculate_match_score(&lower_query, m, &lowered_ids[i]);
            (score >= 3).then_some((i, score))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.cmp(&a.1) // higher score first
            .then_with(|| models[a.0].is_loaded.cmp(&models[b.0].is_loaded).reverse())
            .then_with(|| models[a.0].id.cmp(&models[b.0].id))
    });
    scored.first().map(|(i, _)| &models[*i])
}

fn calculate_match_score(query: &str, model: &ModelMatchView, model_id_lower: &str) -> usize {
    let mut score = 0;

    for q_part in query
        .split(&['-', '_', ':', '.', '/', ' '])
        .filter(|s| s.len() > 1)
    {
        for m_part in model_id_lower
            .split(&['-', '_', ':', '.', '/', ' '])
            .filter(|s| s.len() > 1)
        {
            if q_part == m_part {
                score += q_part.len() * 2;
            } else if m_part.contains(q_part) || q_part.contains(m_part) {
                score += q_part.len().min(m_part.len());
            }
        }
    }

    if model.arch.eq_ignore_ascii_case(query) {
        score += 5;
    }

    if model.model_type == "llm" && (query.contains("chat") || query.contains("instruct")) {
        score += 3;
    }
    if model.model_type == "vlm" && (query.contains("vision") || query.contains("llava")) {
        score += 3;
    }
    if model.model_type == "embeddings" && query.contains("embed") {
        score += 3;
    }

    if model.is_loaded {
        score += 2;
    }

    if model_id_lower.starts_with(query) {
        score += query.len();
    }

    score
}
