> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# List models

> Fetch a list of models and their details



## OpenAPI

````yaml /openapi.yaml get /api/tags
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
  /api/tags:
    get:
      summary: List models
      description: Fetch a list of models and their details
      operationId: list
      responses:
        '200':
          description: List available models
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ListResponse'
              example:
                models:
                  - name: gemma4
                    model: gemma4
                    modified_at: '2025-10-03T23:34:03.409490317-07:00'
                    size: 9608350245
                    digest: >-
                      c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb
                    details:
                      format: gguf
                      family: gemma4
                      families:
                        - gemma4
                      parameter_size: 8.0B
                      quantization_level: Q4_K_M
      x-codeSamples:
        - lang: bash
          label: List models
          source: |
            curl http://localhost:11434/api/tags
components:
  schemas:
    ListResponse:
      type: object
      properties:
        models:
          type: array
          items:
            $ref: '#/components/schemas/ModelSummary'
    ModelSummary:
      type: object
      description: Summary information for a locally available model
      properties:
        name:
          type: string
          description: Model name
        model:
          type: string
          description: Model name
        remote_model:
          type: string
          description: Name of the upstream model, if the model is remote
        remote_host:
          type: string
          description: URL of the upstream Ollama host, if the model is remote
        modified_at:
          type: string
          description: Last modified timestamp in ISO 8601 format
        size:
          type: integer
          description: Total size of the model on disk in bytes
        digest:
          type: string
          description: SHA256 digest identifier of the model contents
        details:
          type: object
          description: Additional information about the model's format and family
          properties:
            format:
              type: string
              description: Model file format (for example `gguf`)
            family:
              type: string
              description: Primary model family (for example `llama`)
            families:
              type: array
              items:
                type: string
              description: All families the model belongs to, when applicable
            parameter_size:
              type: string
              description: Approximate parameter count label (for example `7B`, `13B`)
            quantization_level:
              type: string
              description: Quantization level used (for example `Q4_0`)

````