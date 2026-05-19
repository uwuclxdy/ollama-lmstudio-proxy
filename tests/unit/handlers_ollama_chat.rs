use super::*;
use serde_json::json;

#[test]
fn extracts_think_from_body() {
    let body = json!({ "think": true, "model": "x", "messages": [] });
    let top = make_top_level_params(&body);
    assert!(top.think.is_some());
    assert_eq!(top.think, Some(&json!(true)));
}

#[test]
fn absent_think_gives_none() {
    let body = json!({ "model": "x", "messages": [] });
    let top = make_top_level_params(&body);
    assert!(top.think.is_none());
}

#[test]
fn extracts_logprobs_and_top_logprobs() {
    let body = json!({ "logprobs": true, "top_logprobs": 3 });
    let top = make_top_level_params(&body);
    assert_eq!(top.logprobs, Some(&json!(true)));
    assert_eq!(top.top_logprobs, Some(&json!(3)));
}
