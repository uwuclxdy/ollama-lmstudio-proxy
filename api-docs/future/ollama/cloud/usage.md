> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Usage

> Review usage, balances, and plans for Ollama's cloud models.

Review recent activity on the [usage page](https://ollama.com/settings), or compare available [plans](https://ollama.com/pricing).

## Usage API

Use the [cloud usage endpoint](https://docs.ollama.com/api/cloud-usage.md) to retrieve request counts, token usage, and cost over the last 24 hours, 7 days, or 30 days.

```shell theme={"system"}
curl 'https://ollama.com/api/usage?range=24h' \
  -H "Authorization: Bearer $OLLAMA_API_KEY"
```

New usage may take time to appear.

## Balance API

Use the [balance endpoint](https://docs.ollama.com/api/balance.md) to retrieve your remaining included and purchased credits.

```shell theme={"system"}
curl https://ollama.com/api/balance \
  -H "Authorization: Bearer $OLLAMA_API_KEY"
```

Both endpoints require an [API key](https://docs.ollama.com/cloud/api.md). Team admins can also retrieve usage for the whole team.
