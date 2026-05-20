> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Show model details



## OpenAPI

````yaml /openapi.yaml post /api/show
openapi: 3.1.0
info:
  title: Ollama API
  version: 0.1.0
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT
  description: |
    OpenAPI specification for the Ollama HTTP API
servers:
  - url: http://localhost:11434
    description: Ollama
security: []
paths:
  /api/show:
    post:
      summary: Show model details
      operationId: show
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ShowRequest'
            example:
              model: gemma3
      responses:
        '200':
          description: Model information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ShowResponse'
              example:
                parameters: |-
                  temperature 0.7
                  num_ctx 2048
                license: |-
                  Gemma Terms of Use 

                  Last modified: February 21, 2024...
                capabilities:
                  - completion
                  - vision
                modified_at: '2025-08-14T15:49:43.634137516-07:00'
                details:
                  parent_model: ''
                  format: gguf
                  family: gemma3
                  families:
                    - gemma3
                  parameter_size: 4.3B
                  quantization_level: Q4_K_M
                model_info:
                  gemma3.attention.head_count: 8
                  gemma3.attention.head_count_kv: 4
                  gemma3.attention.key_length: 256
                  gemma3.attention.sliding_window: 1024
                  gemma3.attention.value_length: 256
                  gemma3.block_count: 34
                  gemma3.context_length: 131072
                  gemma3.embedding_length: 2560
                  gemma3.feed_forward_length: 10240
                  gemma3.mm.tokens_per_image: 256
                  gemma3.vision.attention.head_count: 16
                  gemma3.vision.attention.layer_norm_epsilon: 0.000001
                  gemma3.vision.block_count: 27
                  gemma3.vision.embedding_length: 1152
                  gemma3.vision.feed_forward_length: 4304
                  gemma3.vision.image_size: 896
                  gemma3.vision.num_channels: 3
                  gemma3.vision.patch_size: 14
                  general.architecture: gemma3
                  general.file_type: 15
                  general.parameter_count: 4299915632
                  general.quantization_version: 2
                  tokenizer.ggml.add_bos_token: true
                  tokenizer.ggml.add_eos_token: false
                  tokenizer.ggml.add_padding_token: false
                  tokenizer.ggml.add_unknown_token: false
                  tokenizer.ggml.bos_token_id: 2
                  tokenizer.ggml.eos_token_id: 1
                  tokenizer.ggml.merges: null
                  tokenizer.ggml.model: llama
                  tokenizer.ggml.padding_token_id: 0
                  tokenizer.ggml.pre: default
                  tokenizer.ggml.scores: null
                  tokenizer.ggml.token_type: null
                  tokenizer.ggml.tokens: null
                  tokenizer.ggml.unknown_token_id: 3
      x-codeSamples:
        - lang: bash
          label: Default
          source: |
            curl http://localhost:11434/api/show -d '{
              "model": "gemma3"
            }'
        - lang: bash
          label: Verbose
          source: |
            curl http://localhost:11434/api/show -d '{
              "model": "gemma3",
              "verbose": true
            }'
components:
  schemas:
    ShowRequest:
      type: object
      required:
        - model
      properties:
        model:
          type: string
          description: Model name to show
        verbose:
          type: boolean
          description: If true, includes large verbose fields in the response.
    ShowResponse:
      type: object
      properties:
        parameters:
          type: string
          description: Model parameter settings serialized as text
        license:
          type: string
          description: The license of the model
        modified_at:
          type: string
          description: Last modified timestamp in ISO 8601 format
        details:
          type: object
          description: High-level model details
        template:
          type: string
          description: The template used by the model to render prompts
        capabilities:
          type: array
          items:
            type: string
          description: List of supported features
        model_info:
          type: object
          description: Additional model metadata

````