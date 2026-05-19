use super::*;
// handlers/response.rs
//
// handle_response is the only public function; it requires a live
// reqwest::Response (network) so it cannot be unit-tested in isolation.
//
// What we can test:
//   - ResponseContext enum variants construct and match correctly.
//   - ResponseParams struct is constructible (verifies public field names
//     and types match what the source declares).

#[test]
fn response_context_chat_variant() {
    let ctx = ResponseContext::Chat { message_count: 3 };
    let ResponseContext::Chat { message_count } = ctx else {
        panic!("expected Chat variant");
    };
    assert_eq!(message_count, 3);
}

#[test]
fn response_context_generate_variant() {
    let ctx = ResponseContext::Generate {
        prompt: "hello world".to_string(),
    };
    let ResponseContext::Generate { prompt } = ctx else {
        panic!("expected Generate variant");
    };
    assert_eq!(prompt, "hello world");
}
