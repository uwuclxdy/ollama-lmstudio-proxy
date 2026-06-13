> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Pull a model



## OpenAPI

````yaml /openapi.yaml post /api/pull
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
  /api/pull:
    post:
      summary: Pull a model
      operationId: pull
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PullRequest'
            example:
              model: gemma4
      responses:
        '200':
          description: Pull status updates.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StatusResponse'
              example:
                status: success
            application/x-ndjson:
              schema:
                $ref: '#/components/schemas/StatusEvent'
              example:
                status: success
      x-codeSamples:
        - lang: bash
          label: Default
          source: |
            curl http://localhost:11434/api/pull -d '{
              "model": "gemma4"
            }'
        - lang: bash
          label: Non-streaming
          source: |
            curl http://localhost:11434/api/pull -d '{
              "model": "gemma4",
              "stream": false
            }'
components:
  schemas:
    PullRequest:
      type: object
      required:
        - model
      properties:
        model:
          type: string
          description: Name of the model to download
        insecure:
          type: boolean
          description: Allow downloading over insecure connections
        stream:
          type: boolean
          default: true
          description: Stream progress updates
    StatusResponse:
      type: object
      properties:
        status:
          type: string
          description: Current status message
    StatusEvent:
      type: object
      properties:
        status:
          type: string
          description: Human-readable status message
        digest:
          type: string
          description: Content digest associated with the status, if applicable
        total:
          type: integer
          description: Total number of bytes expected for the operation
        completed:
          type: integer
          description: Number of bytes transferred so far

````