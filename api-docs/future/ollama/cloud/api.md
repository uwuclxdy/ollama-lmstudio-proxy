> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Creating an API key

> Create an API key and authenticate requests to Ollama's cloud API.

Create an API key to call cloud models directly through `https://ollama.com/api`. A local Ollama installation is not required.

## Create a key

1. Sign in to [Ollama](https://ollama.com/signin).
2. Open [API keys](https://ollama.com/settings/keys).
3. Create a key and copy it. You will use it as a bearer token with each request.

Set the key as an environment variable in Bash or Zsh:

```shell theme={"system"}
export OLLAMA_API_KEY="your_api_key"
```

Keep API keys out of browser code and source control.

## Send a request

```shell theme={"system"}
curl https://ollama.com/api/chat \
  -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "stream": false
  }'
```

Read the answer from `message.content`.

For classification, yes/no questions, and scoring, call System One at `https://ollama.com/v1/systemone` with the same API key. See the [System One cloud examples](https://docs.ollama.com/capabilities/decision.md#use-a-cloud-model).

You can also use [OpenAI](https://docs.ollama.com/api/openai-compatibility.md) or [Anthropic](https://docs.ollama.com/api/anthropic-compatibility.md) clients. Each supports a subset of the original API. For endpoint schemas and response fields, see the [API Reference](https://docs.ollama.com/api/introduction.md).
