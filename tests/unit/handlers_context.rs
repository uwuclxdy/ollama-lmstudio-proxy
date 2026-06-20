use tempfile::TempDir;

fn fresh_blob_dir() -> TempDir {
    TempDir::new().unwrap()
}

fn fresh_vm_store(dir: &TempDir) -> std::sync::Arc<crate::storage::VirtualModelStore> {
    std::sync::Arc::new(
        crate::storage::VirtualModelStore::load(dir.path().join("vm.json")).unwrap(),
    )
}

fn fresh_blob_store(dir: &TempDir) -> std::sync::Arc<crate::storage::BlobStore> {
    std::sync::Arc::new(crate::storage::BlobStore::new(dir.path()).unwrap())
}

// endpoint_url and append_query_params only touch `lmstudio_url` and the
// passed-in strings — the other fields are never accessed.

macro_rules! with_ctx {
    ($url:expr, |$ctx:ident| $body:expr) => {{
        let client = reqwest::Client::new();
        let vm_dir = fresh_blob_dir();
        let bs_dir = fresh_blob_dir();
        let vms = fresh_vm_store(&vm_dir);
        let bs = fresh_blob_store(&bs_dir);
        let $ctx = crate::api::RequestContext {
            client: &client,
            lmstudio_url: $url,
            virtual_models: vms,
            blob_store: bs,
            load_tracker: crate::model::LoadTracker::new(),
        };
        $body
    }};
}

// --- endpoint_url ---

#[test]
fn endpoint_url_concatenates_base_and_path() {
    with_ctx!("http://localhost:1234", |ctx| {
        assert_eq!(
            ctx.endpoint_url("/v1/chat/completions"),
            "http://localhost:1234/v1/chat/completions"
        );
    });
}

#[test]
fn endpoint_url_empty_endpoint() {
    with_ctx!("http://localhost:1234", |ctx| {
        assert_eq!(ctx.endpoint_url(""), "http://localhost:1234");
    });
}

#[test]
fn endpoint_url_base_without_trailing_slash() {
    with_ctx!("http://host", |ctx| {
        assert_eq!(ctx.endpoint_url("/path"), "http://host/path");
    });
}

// --- append_query_params ---

#[test]
fn append_query_params_none_leaves_url_unchanged() {
    with_ctx!("http://x", |ctx| {
        let url = "http://x/v1/models".to_string();
        assert_eq!(ctx.append_query_params(url.clone(), None), url);
    });
}

#[test]
fn append_query_params_adds_question_mark_when_no_query() {
    with_ctx!("http://x", |ctx| {
        let result = ctx.append_query_params("http://x/v1/models".to_string(), Some("foo=bar"));
        assert_eq!(result, "http://x/v1/models?foo=bar");
    });
}

#[test]
fn append_query_params_adds_ampersand_when_query_exists() {
    with_ctx!("http://x", |ctx| {
        let result =
            ctx.append_query_params("http://x/v1/models?existing=1".to_string(), Some("new=2"));
        assert_eq!(result, "http://x/v1/models?existing=1&new=2");
    });
}

#[test]
fn append_query_params_multiple_params_in_one_string() {
    with_ctx!("http://x", |ctx| {
        let result = ctx.append_query_params("http://x/v1".to_string(), Some("a=1&b=2"));
        assert_eq!(result, "http://x/v1?a=1&b=2");
    });
}

#[test]
fn append_query_params_does_not_mutate_url_when_none() {
    with_ctx!("http://x", |ctx| {
        let url = "http://x/api?already=set".to_string();
        assert_eq!(ctx.append_query_params(url.clone(), None), url);
    });
}
