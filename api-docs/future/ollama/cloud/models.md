> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Models

> Run Ollama's cloud models from the app, CLI, local API, or official libraries.

Cloud models run in Ollama's cloud while remaining available to local tools and integrations through Ollama.

## Sign in

[Download Ollama](https://ollama.com/download), then open the app or sign in from the CLI:

```shell theme={"system"}
ollama signin
```

Browse available models in the [cloud model library](https://ollama.com/search?c=cloud).

## Run a cloud model

Cloud models do not need to be downloaded. Run one from the CLI:

```shell theme={"system"}
ollama run gemma4:cloud
```

You can also open the Ollama app to select a cloud model or connect an integration. See the [quickstart](https://docs.ollama.com/quickstart.md) for setup.

## Model names

For direct requests to `ollama.com`, list models with:

```shell theme={"system"}
curl https://ollama.com/api/tags
```

Use the name returned by this list, such as `gemma4:31b`. In the Ollama app, CLI, official libraries, or local API, use the `:cloud` name, such as `gemma4:cloud`.

To call a model without installing Ollama, first [create an API key](https://docs.ollama.com/cloud/api.md).

## System One

Use [Clef](https://ollama.com/library/clef) for classification, yes/no questions, and scoring in the cloud. Through a signed-in local Ollama server, use `clef:cloud` with Ollama v0.40.5 or later. Direct API requests use `clef`. Cloud access requires a paid plan or purchased usage credits. See the [System One cloud examples](https://docs.ollama.com/capabilities/decision.md#use-a-cloud-model).
