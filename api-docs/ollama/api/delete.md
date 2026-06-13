> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Delete a model



## OpenAPI

````yaml /openapi.yaml delete /api/delete
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
  /api/delete:
    delete:
      summary: Delete a model
      operationId: delete
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DeleteRequest'
            example:
              model: gemma4
      responses:
        '200':
          description: Model successfully deleted
      x-codeSamples:
        - lang: bash
          label: Delete model
          source: |
            curl -X DELETE http://localhost:11434/api/delete -d '{
              "model": "gemma4"
            }'
components:
  schemas:
    DeleteRequest:
      type: object
      required:
        - model
      properties:
        model:
          type: string
          description: Model name to delete

````