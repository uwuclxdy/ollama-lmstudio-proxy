> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Copy a model



## OpenAPI

````yaml /openapi.yaml post /api/copy
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
  /api/copy:
    post:
      summary: Copy a model
      operationId: copy
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CopyRequest'
            example:
              source: gemma4
              destination: gemma4-backup
      responses:
        '200':
          description: Model successfully copied
      x-codeSamples:
        - lang: bash
          label: Copy a model to a new name
          source: |
            curl http://localhost:11434/api/copy -d '{
              "source": "gemma4",
              "destination": "gemma4-backup"
            }'
components:
  schemas:
    CopyRequest:
      type: object
      required:
        - source
        - destination
      properties:
        source:
          type: string
          description: Existing model name to copy from
        destination:
          type: string
          description: New model name to create

````