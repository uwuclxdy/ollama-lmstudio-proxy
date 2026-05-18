//! Tests for deterministic model-name resolution.
//!
//! Ollama clients send `model:tag` strings; the proxy must map them to a single,
//! deterministic LM Studio model id. When multiple substring matches exist,
//! resolution must:
//!   1. prefer an exact match (case-insensitive)
//!   2. otherwise prefer a model that is currently loaded (cheap to query)
//!   3. otherwise prefer the model whose id is closest in length to the query
//!      (most specific match), with ties broken by lexicographic order on id
//!
//! Failing those criteria, no result must come from non-deterministic iteration
//! order: identical input must always produce identical output.

#[path = "../src/model/matcher.rs"]
#[allow(dead_code)]
mod matcher;

use matcher::{ModelMatchView, find_best_match};

fn mv(id: &str, loaded: bool) -> ModelMatchView {
    ModelMatchView {
        id: id.to_string(),
        arch: String::new(),
        model_type: "llm".to_string(),
        is_loaded: loaded,
    }
}

#[test]
fn exact_match_wins_over_substring() {
    let models = vec![
        mv("qwen2-7b-instruct", false),
        mv("qwen2-7b", false),
        mv("qwen2-7b-chat", false),
    ];
    let result = find_best_match("qwen2-7b", &models).expect("should match");
    assert_eq!(result.id, "qwen2-7b");
}

#[test]
fn ambiguous_substring_match_is_deterministic_loaded_wins() {
    // Two equally plausible substring candidates; the loaded one must win.
    let models = vec![mv("qwen2-7b-instruct", false), mv("qwen2-7b-chat", true)];
    let result = find_best_match("qwen2-7b", &models).expect("should match");
    assert_eq!(
        result.id, "qwen2-7b-chat",
        "loaded model must be preferred when multiple substring candidates exist"
    );
}

#[test]
fn ambiguous_substring_match_is_deterministic_shorter_wins() {
    // Neither loaded; the model whose id is closest in length to the query wins.
    // "qwen2-7b" has length 8; "qwen2-7b-chat" (13) is closer than "qwen2-7b-instruct-v0.2" (22).
    let models = vec![
        mv("qwen2-7b-instruct-v0.2", false),
        mv("qwen2-7b-chat", false),
    ];
    let result = find_best_match("qwen2-7b", &models).expect("should match");
    assert_eq!(result.id, "qwen2-7b-chat");
}

#[test]
fn ambiguous_substring_match_tiebreaks_alphabetically() {
    // Both same length, neither loaded — alphabetical order.
    let models = vec![
        mv("qwen2-7b-instruct", false),
        mv("qwen2-7b-chat-out", false),
    ];
    let result = find_best_match("qwen2-7b", &models).expect("should match");
    // "qwen2-7b-chat-out" sorts before "qwen2-7b-instruct"
    assert_eq!(result.id, "qwen2-7b-chat-out");
}

#[test]
fn results_are_stable_across_input_order_permutations() {
    let base = vec![
        mv("qwen2-7b-instruct", true),
        mv("qwen2-7b-chat", false),
        mv("qwen2-7b-tools", false),
    ];
    let reversed: Vec<_> = base.iter().rev().cloned().collect();
    let r1 = find_best_match("qwen2-7b", &base).unwrap().id.clone();
    let r2 = find_best_match("qwen2-7b", &reversed).unwrap().id.clone();
    assert_eq!(r1, r2, "matcher must be deterministic across input order");
}

#[test]
fn no_match_returns_none() {
    let models = vec![mv("granite-3.0-2b", false)];
    let result = find_best_match("nonexistent-model", &models);
    assert!(result.is_none());
}

#[test]
fn case_insensitive_exact_match() {
    let models = vec![mv("Qwen2-7B-Instruct", false)];
    let result = find_best_match("qwen2-7b-instruct", &models).expect("should match");
    assert_eq!(result.id, "Qwen2-7B-Instruct");
}
