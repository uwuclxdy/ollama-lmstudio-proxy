# 🧰 MCP integrations

Available only on the native chat path. Start the proxy with `--use-native-chat`, or with `--native-chat-streaming` if you only need it on streaming requests (see [Configuration](Configuration)).

On that path `/api/chat` accepts an `integrations` array, forwarded verbatim to LM Studio's `/api/v1/chat`. Non-array values are ignored.

## Element types

Each element of the array may be:

- A bare plugin-id string: `"huggingface"`
- A plugin object: `{"type": "plugin", "id": "browser", "allowed_tools": ["browser_navigate"]}`
- An ephemeral MCP server: `{"type": "ephemeral_mcp", "server_label": "hf", "server_url": "https://hf.co/mcp", "allowed_tools": ["model_search"], "headers": {}}`

## LM Studio settings

Enable the matching options under **Developer** in the LM Studio UI:

- **Allow calling servers from mcp.json** enables `plugin` entries.
- **Allow per-request MCPs** enables `ephemeral_mcp` entries.

Overrides such as `stop` or `seed` ride alongside these untouched.
