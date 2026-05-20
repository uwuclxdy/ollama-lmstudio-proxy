use super::*;
use serde_json::json;
use tempfile::TempDir;

fn make_store(dir: &TempDir) -> VirtualModelStore {
    let path = dir.path().join("virtual_models.json");
    VirtualModelStore::load(path).expect("load should succeed")
}

fn default_metadata() -> VirtualModelMetadata {
    VirtualModelMetadata::default()
}

// --- build_metadata_from_request ---

#[test]
fn build_metadata_empty_body_no_base() {
    let meta = VirtualModelStore::build_metadata_from_request(&json!({}), None);
    assert!(meta.system_prompt.is_none());
    assert!(meta.template.is_none());
    assert!(meta.parameters.is_none());
    assert!(meta.license.is_none());
    assert!(meta.adapters.is_none());
    assert!(meta.messages.is_none());
}

#[test]
fn build_metadata_all_fields_populated() {
    let body = json!({
        "system": "be concise",
        "template": "{{ .Prompt }}",
        "parameters": {"temperature": 0.7},
        "license": "MIT",
        "adapters": [],
        "messages": [{"role": "user", "content": "hi"}]
    });
    let meta = VirtualModelStore::build_metadata_from_request(&body, None);
    assert_eq!(meta.system_prompt.as_deref(), Some("be concise"));
    assert_eq!(meta.template.as_deref(), Some("{{ .Prompt }}"));
    assert!(meta.parameters.is_some());
    assert!(meta.license.is_some());
    assert!(meta.adapters.is_some());
    assert_eq!(meta.messages.as_ref().map(|m| m.len()), Some(1));
}

#[test]
fn build_metadata_base_preserved_when_body_empty() {
    let base = VirtualModelMetadata {
        system_prompt: Some("inherited".to_string()),
        ..VirtualModelMetadata::default()
    };
    let meta = VirtualModelStore::build_metadata_from_request(&json!({}), Some(base));
    assert_eq!(meta.system_prompt.as_deref(), Some("inherited"));
}

#[test]
fn build_metadata_body_overrides_base() {
    let base = VirtualModelMetadata {
        system_prompt: Some("old".to_string()),
        ..VirtualModelMetadata::default()
    };
    let body = json!({"system": "new"});
    let meta = VirtualModelStore::build_metadata_from_request(&body, Some(base));
    assert_eq!(meta.system_prompt.as_deref(), Some("new"));
}

#[test]
fn build_metadata_messages_non_array_not_set() {
    let body = json!({"messages": "not-an-array"});
    let meta = VirtualModelStore::build_metadata_from_request(&body, None);
    assert!(meta.messages.is_none());
}

// --- VirtualModelEntry fields ---

#[test]
fn entry_fields_are_set_correctly() {
    let now = Utc::now();
    let entry = VirtualModelEntry {
        name: "mymodel".to_string(),
        source_model: "llama3".to_string(),
        target_model_id: "llama-3-8b".to_string(),
        created_at: now,
        updated_at: now,
        metadata: default_metadata(),
    };
    assert_eq!(entry.name, "mymodel");
    assert_eq!(entry.source_model, "llama3");
    assert_eq!(entry.target_model_id, "llama-3-8b");
    assert_eq!(entry.created_at, now);
}

// --- VirtualModelStore::load ---

#[test]
fn load_missing_file_returns_empty_store() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let entries = rt.block_on(store.list());
    assert!(entries.is_empty());
}

#[test]
fn load_creates_parent_directory() {
    let dir = TempDir::new().unwrap();
    let nested_path = dir.path().join("nested/dir/models.json");
    let store = VirtualModelStore::load(nested_path.clone()).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert!(rt.block_on(store.list()).is_empty());
}

#[test]
fn load_corrupt_json_returns_empty_store() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vm.json");
    std::fs::write(&path, b"not valid json at all!!!").unwrap();
    let store = VirtualModelStore::load(path).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert!(rt.block_on(store.list()).is_empty());
}

#[test]
fn load_empty_file_returns_empty_store() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vm.json");
    std::fs::write(&path, b"").unwrap();
    let store = VirtualModelStore::load(path).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert!(rt.block_on(store.list()).is_empty());
}

// --- create_alias, upsert_alias, get, list, delete ---

#[tokio::test]
async fn create_alias_and_get() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .create_alias(
            "myalias",
            "llama3".to_string(),
            "llama-3-8b".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    let entry = store.get("myalias").await.unwrap();
    assert_eq!(entry.name, "myalias");
    assert_eq!(entry.source_model, "llama3");
    assert_eq!(entry.target_model_id, "llama-3-8b");
}

#[tokio::test]
async fn get_missing_returns_none() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);
    assert!(store.get("nonexistent").await.is_none());
}

#[tokio::test]
async fn duplicate_alias_returns_error() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .create_alias(
            "alias",
            "src".to_string(),
            "tgt".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    let err = store
        .create_alias(
            "alias",
            "src2".to_string(),
            "tgt2".to_string(),
            default_metadata(),
        )
        .await
        .unwrap_err();

    assert!(err.message.contains("already exists"));
}

#[tokio::test]
async fn delete_existing_removes_entry() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .create_alias(
            "to_del",
            "s".to_string(),
            "t".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    store.delete("to_del").await.unwrap();
    assert!(store.get("to_del").await.is_none());
}

#[tokio::test]
async fn delete_missing_returns_error() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);
    let err = store.delete("ghost").await.unwrap_err();
    assert!(err.message.contains("not managed by proxy") || err.message.contains("ghost"));
}

#[tokio::test]
async fn list_returns_all_entries() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .create_alias("a1", "s".to_string(), "t1".to_string(), default_metadata())
        .await
        .unwrap();
    store
        .create_alias("a2", "s".to_string(), "t2".to_string(), default_metadata())
        .await
        .unwrap();

    let all = store.list().await;
    assert_eq!(all.len(), 2);
}

// --- upsert_alias ---

#[tokio::test]
async fn upsert_alias_creates_when_absent() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .upsert_alias(
            "newalias",
            "src".to_string(),
            "tgt".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    let entry = store.get("newalias").await.unwrap();
    assert_eq!(entry.target_model_id, "tgt");
}

#[tokio::test]
async fn upsert_alias_overwrites_existing() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .upsert_alias(
            "alias",
            "src-old".to_string(),
            "tgt-old".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    store
        .upsert_alias(
            "alias",
            "src-new".to_string(),
            "tgt-new".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    let entry = store.get("alias").await.unwrap();
    assert_eq!(entry.source_model, "src-new");
    assert_eq!(entry.target_model_id, "tgt-new");
}

#[tokio::test]
async fn upsert_alias_preserves_created_at_on_overwrite() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .upsert_alias(
            "alias",
            "src".to_string(),
            "tgt".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();
    let original_created_at = store.get("alias").await.unwrap().created_at;

    store
        .upsert_alias(
            "alias",
            "src2".to_string(),
            "tgt2".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();
    let updated = store.get("alias").await.unwrap();

    assert_eq!(
        updated.created_at, original_created_at,
        "created_at must be preserved across overwrites"
    );
    assert!(
        updated.updated_at >= original_created_at,
        "updated_at must advance"
    );
}

// --- persistence round-trip ---

#[tokio::test]
async fn persistence_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vm.json");

    {
        let store = VirtualModelStore::load(path.clone()).unwrap();
        store
            .create_alias(
                "persisted",
                "src".to_string(),
                "tgt-id".to_string(),
                default_metadata(),
            )
            .await
            .unwrap();
    }

    let store2 = VirtualModelStore::load(path).unwrap();
    let entry = store2.get("persisted").await.unwrap();
    assert_eq!(entry.target_model_id, "tgt-id");
}

// --- canonical name normalization ---

#[tokio::test]
async fn get_strips_latest_tag_to_match_stored_alias() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);

    store
        .create_alias(
            "llama3",
            "s".to_string(),
            "t".to_string(),
            default_metadata(),
        )
        .await
        .unwrap();

    let entry = store
        .get("llama3:latest")
        .await
        .expect("':latest' must canonicalize to bare name");
    assert_eq!(entry.name, "llama3");
}
