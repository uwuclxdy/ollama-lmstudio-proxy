> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Generate embeddings

> Creates vector embeddings representing the input text



## OpenAPI

````yaml /openapi.yaml post /api/embed
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
  /api/embed:
    post:
      summary: Generate embeddings
      description: Creates vector embeddings representing the input text
      operationId: embed
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EmbedRequest'
            example:
              model: embeddinggemma
              input: Generate embeddings for this text
      responses:
        '200':
          description: Vector embeddings for the input text
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmbedResponse'
              example:
                model: embeddinggemma
                embeddings:
                  - - 0.010071029
                    - -0.0017594862
                    - 0.05007221
                    - 0.04692972
                    - 0.054916814
                    - 0.008599704
                    - 0.105441414
                    - -0.025878139
                    - 0.12958129
                    - 0.031952348
                total_duration: 14143917
                load_duration: 1019500
                prompt_eval_count: 8
      x-codeSamples:
        - lang: bash
          label: Default
          source: |
            curl http://localhost:11434/api/embed -d '{
              "model": "embeddinggemma",
              "input": "Why is the sky blue?"
            }'
        - lang: bash
          label: Multiple inputs
          source: |
            curl http://localhost:11434/api/embed -d '{
              "model": "embeddinggemma",
              "input": [
                "Why is the sky blue?",
                "Why is the grass green?"
              ]
            }'
        - lang: bash
          label: Truncation
          source: |
            curl http://localhost:11434/api/embed -d '{
              "model": "embeddinggemma",
              "input": "Generate embeddings for this text",
              "truncate": true
            }'
        - lang: bash
          label: Dimensions
          source: |
            curl http://localhost:11434/api/embed -d '{
              "model": "embeddinggemma",
              "input": "Generate embeddings for this text",
              "dimensions": 128
            }'
components:
  schemas:
    EmbedRequest:
      type: object
      required:
        - model
        - input
      properties:
        model:
          type: string
          description: Model name
        input:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
          description: Text or array of texts to generate embeddings for
        truncate:
          type: boolean
          default: true
          description: >-
            If true, truncate inputs that exceed the context window. If false,
            returns an error.
        dimensions:
          type: integer
          description: Number of dimensions to generate embeddings for
        keep_alive:
          type: string
          description: Model keep-alive duration
        options:
          $ref: '#/components/schemas/ModelOptions'
    EmbedResponse:
      type: object
      properties:
        model:
          type: string
          description: Model that produced the embeddings
        embeddings:
          type: array
          items:
            type: array
            items:
              type: number
          description: Array of vector embeddings
        total_duration:
          type: integer
          description: Total time spent generating in nanoseconds
        load_duration:
          type: integer
          description: Load time in nanoseconds
        prompt_eval_count:
          type: integer
          description: Number of input tokens processed to generate embeddings
    ModelOptions:
      type: object
      description: Runtime options that control text generation
      properties:
        seed:
          type: integer
          description: Random seed used for reproducible outputs
        temperature:
          type: number
          format: float
          description: Controls randomness in generation (higher = more random)
        top_k:
          type: integer
          description: Limits next token selection to the K most likely
        top_p:
          type: number
          format: float
          description: Cumulative probability threshold for nucleus sampling
        min_p:
          type: number
          format: float
          description: Minimum probability threshold for token selection
        stop:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
          description: Stop sequences that will halt generation
        num_ctx:
          type: integer
          description: Context length size (number of tokens)
        num_predict:
          type: integer
          description: Maximum number of tokens to generate
      additionalProperties: true

````