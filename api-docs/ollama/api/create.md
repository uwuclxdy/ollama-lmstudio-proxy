> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Create a model



## OpenAPI

````yaml /openapi.yaml post /api/create
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
  /api/create:
    post:
      summary: Create a model
      operationId: create
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateRequest'
            example:
              model: mario
              from: gemma4
              system: You are Mario from Super Mario Bros.
      responses:
        '200':
          description: Stream of create status updates
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
            curl http://localhost:11434/api/create -d '{
              "from": "gemma4",
              "model": "alpaca",
              "system": "You are Alpaca, a helpful AI assistant. You only answer with Emojis."
            }'
        - lang: bash
          label: Create from existing
          source: |
            curl http://localhost:11434/api/create -d '{
              "model": "ollama",
              "from": "gemma4",
              "system": "You are Ollama the llama."
            }'
        - lang: bash
          label: Quantize
          source: |
            curl http://localhost:11434/api/create -d '{
              "model": "llama3.1:8b-instruct-Q4_K_M",
              "from": "llama3.1:8b-instruct-fp16",
              "quantize": "q4_K_M"
            }'
components:
  schemas:
    CreateRequest:
      type: object
      required:
        - model
      properties:
        model:
          type: string
          description: Name for the model to create
        from:
          type: string
          description: Existing model to create from
        template:
          type: string
          description: Prompt template to use for the model
        license:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
          description: License string or list of licenses for the model
        system:
          type: string
          description: System prompt to embed in the model
        parameters:
          type: object
          description: Key-value parameters for the model
        messages:
          description: Message history to use for the model
          type: array
          items:
            $ref: '#/components/schemas/ChatMessage'
        quantize:
          type: string
          description: Quantization level to apply (e.g. `q4_K_M`, `q8_0`)
        stream:
          type: boolean
          default: true
          description: Stream status updates
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
    ChatMessage:
      type: object
      required:
        - role
        - content
      properties:
        role:
          type: string
          enum:
            - system
            - user
            - assistant
            - tool
          description: Author of the message.
        content:
          type: string
          description: Message text content
        images:
          type: array
          items:
            type: string
            description: Base64-encoded image content
          description: Optional list of inline images for multimodal models
        tool_calls:
          type: array
          items:
            $ref: '#/components/schemas/ToolCall'
          description: Tool call requests produced by the model
    ToolCall:
      type: object
      properties:
        function:
          type: object
          required:
            - name
          properties:
            name:
              type: string
              description: Name of the function to call
            description:
              type: string
              description: What the function does
            arguments:
              type: object
              description: JSON object of arguments to pass to the function

````