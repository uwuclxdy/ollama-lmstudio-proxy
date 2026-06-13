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
              model: gemma4
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
                  family: gemma4
                  families:
                    - gemma4
                  parameter_size: 8.0B
                  quantization_level: Q4_K_M
                model_info:
                  gemma4.attention.head_count: 8
                  gemma4.attention.head_count_kv: 2
                  gemma4.attention.key_length: 512
                  gemma4.attention.key_length_swa: 256
                  gemma4.attention.layer_norm_rms_epsilon: 0.000001
                  gemma4.attention.shared_kv_layers: 18
                  gemma4.attention.sliding_window: 512
                  gemma4.attention.value_length: 512
                  gemma4.attention.value_length_swa: 256
                  gemma4.audio.attention.head_count: 8
                  gemma4.audio.attention.layer_norm_epsilon: 0.000001
                  gemma4.audio.block_count: 12
                  gemma4.audio.conv_kernel_size: 5
                  gemma4.audio.embedding_length: 1024
                  gemma4.audio.feed_forward_length: 4096
                  gemma4.block_count: 42
                  gemma4.context_length: 131072
                  gemma4.embedding_length: 2560
                  gemma4.embedding_length_per_layer_input: 256
                  gemma4.feed_forward_length: 10240
                  gemma4.final_logit_softcapping: 30
                  gemma4.rope.dimension_count: 512
                  gemma4.rope.dimension_count_swa: 256
                  gemma4.rope.freq_base: 1000000
                  gemma4.rope.freq_base_swa: 10000
                  gemma4.vision.attention.head_count: 12
                  gemma4.vision.attention.layer_norm_epsilon: 0.000001
                  gemma4.vision.block_count: 16
                  gemma4.vision.embedding_length: 768
                  gemma4.vision.feed_forward_length: 3072
                  gemma4.vision.num_channels: 3
                  gemma4.vision.patch_size: 16
                  gemma4.vision.projector.scale_factor: 3
                  general.architecture: gemma4
                  general.file_type: 15
                  general.quantization_version: 2
                  tokenizer.ggml.add_bos_token: false
                  tokenizer.ggml.add_eos_token: false
                  tokenizer.ggml.add_mask_token: false
                  tokenizer.ggml.add_padding_token: false
                  tokenizer.ggml.add_unknown_token: false
                  tokenizer.ggml.bos_token_id: 2
                  tokenizer.ggml.eos_token_id: 1
                  tokenizer.ggml.eos_token_ids:
                    - 1
                    - 106
                    - 50
                  tokenizer.ggml.mask_token_id: 4
                  tokenizer.ggml.merges: null
                  tokenizer.ggml.model: llama
                  tokenizer.ggml.padding_token_id: 0
                  tokenizer.ggml.pre: gemma4
                  tokenizer.ggml.scores: null
                  tokenizer.ggml.token_type: null
                  tokenizer.ggml.tokens: null
                  tokenizer.ggml.unknown_token_id: 3
      x-codeSamples:
        - lang: bash
          label: Default
          source: |
            curl http://localhost:11434/api/show -d '{
              "model": "gemma4"
            }'
        - lang: bash
          label: Verbose
          source: |
            curl http://localhost:11434/api/show -d '{
              "model": "gemma4",
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