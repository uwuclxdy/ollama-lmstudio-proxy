> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# List running models

> Retrieve a list of models that are currently running



## OpenAPI

````yaml /openapi.yaml get /api/ps
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
  /api/ps:
    get:
      summary: List running models
      description: Retrieve a list of models that are currently running
      operationId: ps
      responses:
        '200':
          description: Models currently loaded into memory
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PsResponse'
              example:
                models:
                  - name: gemma3
                    model: gemma3
                    size: 6591830464
                    digest: >-
                      a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a
                    details:
                      parent_model: ''
                      format: gguf
                      family: gemma3
                      families:
                        - gemma3
                      parameter_size: 4.3B
                      quantization_level: Q4_K_M
                    expires_at: '2025-10-17T16:47:07.93355-07:00'
                    size_vram: 5333539264
                    context_length: 4096
      x-codeSamples:
        - lang: bash
          label: List running models
          source: |
            curl http://localhost:11434/api/ps
components:
  schemas:
    PsResponse:
      type: object
      properties:
        models:
          type: array
          items:
            $ref: '#/components/schemas/Ps'
          description: Currently running models
    Ps:
      type: object
      properties:
        name:
          type: string
          description: Name of the running model
        model:
          type: string
          description: Name of the running model
        size:
          type: integer
          description: Size of the model in bytes
        digest:
          type: string
          description: SHA256 digest of the model
        details:
          type: object
          description: Model details such as format and family
        expires_at:
          type: string
          description: Time when the model will be unloaded
        size_vram:
          type: integer
          description: VRAM usage in bytes
        context_length:
          type: integer
          description: Context length for the running model

````