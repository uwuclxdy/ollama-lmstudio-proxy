use super::*;
use serde_json::json;

#[test]
fn lifts_top_level_truncate_into_options() {
    let mut body = json!({
        "model": "all-minilm",
        "input": "hello",
        "truncate": false
    });
    lift_embed_top_level_params(&mut body);
    assert_eq!(
        body.pointer("/options/truncate"),
        Some(&json!(false)),
        "expected truncate to be lifted into options, got {}",
        body
    );
}

#[test]
fn lifts_top_level_dimensions_into_options() {
    let mut body = json!({
        "model": "all-minilm",
        "input": "hello",
        "dimensions": 256
    });
    lift_embed_top_level_params(&mut body);
    assert_eq!(
        body.pointer("/options/dimensions"),
        Some(&json!(256)),
        "expected dimensions to be lifted into options, got {}",
        body
    );
}

#[test]
fn existing_options_takes_precedence_over_top_level() {
    // If caller sends both, the `options` value wins (treat top-level as fallback).
    let mut body = json!({
        "model": "all-minilm",
        "input": "hello",
        "truncate": false,
        "options": { "truncate": true }
    });
    lift_embed_top_level_params(&mut body);
    assert_eq!(body.pointer("/options/truncate"), Some(&json!(true)));
}

#[test]
fn no_top_level_params_leaves_body_unchanged() {
    let mut body = json!({
        "model": "all-minilm",
        "input": "hello",
        "options": { "temperature": 0.5 }
    });
    let before = body.clone();
    lift_embed_top_level_params(&mut body);
    assert_eq!(body, before);
}

#[test]
fn lifts_both_truncate_and_dimensions_creating_options() {
    let mut body = json!({
        "model": "all-minilm",
        "input": ["a", "b"],
        "truncate": true,
        "dimensions": 1024
    });
    lift_embed_top_level_params(&mut body);
    assert_eq!(body.pointer("/options/truncate"), Some(&json!(true)));
    assert_eq!(body.pointer("/options/dimensions"), Some(&json!(1024)));
}
