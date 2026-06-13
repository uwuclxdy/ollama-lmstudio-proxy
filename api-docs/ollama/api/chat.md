> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Generate a chat message

> Generate the next chat message in a conversation between a user and an assistant.



## OpenAPI

````yaml /openapi.yaml post /api/chat
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
  /api/chat:
    post:
      summary: Generate a chat message
      description: >-
        Generate the next chat message in a conversation between a user and an
        assistant.
      operationId: chat
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatRequest'
      responses:
        '200':
          description: Chat response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatResponse'
              example:
                model: gemma3
                created_at: '2025-10-17T23:14:07.414671Z'
                message:
                  role: assistant
                  content: Hello! How can I help you today?
                done: true
                done_reason: stop
                total_duration: 174560334
                load_duration: 101397084
                prompt_eval_count: 11
                prompt_eval_duration: 13074791
                eval_count: 18
                eval_duration: 52479709
            application/x-ndjson:
              schema:
                $ref: '#/components/schemas/ChatStreamEvent'
      x-codeSamples:
        - lang: bash
          label: Default
          source: |
            curl http://localhost:11434/api/chat -d '{
              "model": "gemma3",
              "messages": [
                {
                  "role": "user",
                  "content": "why is the sky blue?"
                }
              ]
            }'
        - lang: bash
          label: Non-streaming
          source: |
            curl http://localhost:11434/api/chat -d '{
              "model": "gemma3",
              "messages": [
                {
                  "role": "user",
                  "content": "why is the sky blue?"
                }
              ],
              "stream": false
            }'
        - lang: bash
          label: Structured outputs
          source: >
            curl -X POST http://localhost:11434/api/chat -H "Content-Type:
            application/json" -d '{
              "model": "gemma3",
              "messages": [
                {
                  "role": "user",
                  "content": "What are the populations of the United States and Canada?"
                }
              ],
              "stream": false,
              "format": {
                "type": "object",
                "properties": {
                  "countries": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "country": {"type": "string"},
                        "population": {"type": "integer"}
                      },
                      "required": ["country", "population"]
                    }
                  }
                },
                "required": ["countries"]
              }
            }'
        - lang: bash
          label: Tool calling
          source: |
            curl http://localhost:11434/api/chat -d '{
              "model": "qwen3",
              "messages": [
                {
                  "role": "user",
                  "content": "What is the weather today in Paris?"
                }
              ],
              "stream": false,
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "location": {
                          "type": "string",
                          "description": "The location to get the weather for, e.g. San Francisco, CA"
                        },
                        "format": {
                          "type": "string",
                          "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                          "enum": ["celsius", "fahrenheit"]
                        }
                      },
                      "required": ["location", "format"]
                    }
                  }
                }
              ]
            }'
        - lang: bash
          label: Thinking
          source: |
            curl http://localhost:11434/api/chat -d '{
              "model": "gpt-oss",
              "messages": [
                {
                  "role": "user",
                  "content": "What is 1+1?"
                }
              ],
              "think": "low"
            }'
        - lang: bash
          label: Images
          source: |
            curl http://localhost:11434/api/chat -d '{
              "model": "gemma3",
              "messages": [
                {
                  "role": "user",
                  "content": "What is in this image?",
                  "images": [
                    "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"
                  ]
                }
              ]
            }'
components:
  schemas:
    ChatRequest:
      type: object
      required:
        - model
        - messages
      properties:
        model:
          type: string
          description: Model name
        messages:
          type: array
          description: >-
            Chat history as an array of message objects (each with a role and
            content)
          items:
            $ref: '#/components/schemas/ChatMessage'
        tools:
          type: array
          description: Optional list of function tools the model may call during the chat
          items:
            $ref: '#/components/schemas/ToolDefinition'
        format:
          oneOf:
            - type: string
              enum:
                - json
            - type: object
          description: Format to return a response in. Can be `json` or a JSON schema
        options:
          $ref: '#/components/schemas/ModelOptions'
        stream:
          type: boolean
          default: true
        think:
          oneOf:
            - type: boolean
            - type: string
              enum:
                - high
                - medium
                - low
          description: >-
            When true, returns separate thinking output in addition to content.
            Can be a boolean (true/false) or a string ("high", "medium", "low")
            for supported models.
        keep_alive:
          oneOf:
            - type: string
            - type: number
          description: >-
            Model keep-alive duration (for example `5m` or `0` to unload
            immediately)
        logprobs:
          type: boolean
          description: Whether to return log probabilities of the output tokens
        top_logprobs:
          type: integer
          description: >-
            Number of most likely tokens to return at each token position when
            logprobs are enabled
    ChatResponse:
      type: object
      properties:
        model:
          type: string
          description: Model name used to generate this message
        created_at:
          type: string
          format: date-time
          description: Timestamp of response creation (ISO 8601)
        message:
          type: object
          properties:
            role:
              type: string
              enum:
                - assistant
              description: Always `assistant` for model responses
            content:
              type: string
              description: Assistant message text
            thinking:
              type: string
              description: Optional deliberate thinking trace when `think` is enabled
            tool_calls:
              type: array
              items:
                $ref: '#/components/schemas/ToolCall'
              description: Tool calls requested by the assistant
            images:
              type: array
              items:
                type: string
              description: Optional base64-encoded images in the response
        done:
          type: boolean
          description: Indicates whether the chat response has finished
        done_reason:
          type: string
          description: Reason the response finished
        total_duration:
          type: integer
          description: Total time spent generating in nanoseconds
        load_duration:
          type: integer
          description: Time spent loading the model in nanoseconds
        prompt_eval_count:
          type: integer
          description: Number of tokens in the prompt
        prompt_eval_duration:
          type: integer
          description: Time spent evaluating the prompt in nanoseconds
        eval_count:
          type: integer
          description: Number of tokens generated in the response
        eval_duration:
          type: integer
          description: Time spent generating tokens in nanoseconds
        logprobs:
          type: array
          items:
            $ref: '#/components/schemas/Logprob'
          description: >-
            Log probability information for the generated tokens when logprobs
            are enabled
    ChatStreamEvent:
      type: object
      properties:
        model:
          type: string
          description: Model name used for this stream event
        created_at:
          type: string
          format: date-time
          description: When this chunk was created (ISO 8601)
        message:
          type: object
          properties:
            role:
              type: string
              description: Role of the message for this chunk
            content:
              type: string
              description: Partial assistant message text
            thinking:
              type: string
              description: Partial thinking text when `think` is enabled
            tool_calls:
              type: array
              items:
                $ref: '#/components/schemas/ToolCall'
              description: Partial tool calls, if any
            images:
              type: array
              items:
                type: string
              description: Partial base64-encoded images, when present
        done:
          type: boolean
          description: True for the final event in the stream
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
    ToolDefinition:
      type: object
      required:
        - type
        - function
      properties:
        type:
          type: string
          enum:
            - function
          description: Type of tool (always `function`)
        function:
          type: object
          required:
            - name
            - parameters
          properties:
            name:
              type: string
              description: Function name exposed to the model
            description:
              type: string
              description: Human-readable description of the function
            parameters:
              type: object
              description: JSON Schema for the function parameters
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
    Logprob:
      type: object
      description: Log probability information for a generated token
      properties:
        token:
          type: string
          description: The text representation of the token
        logprob:
          type: number
          description: The log probability of this token
        bytes:
          type: array
          items:
            type: integer
          description: The raw byte representation of the token
        top_logprobs:
          type: array
          items:
            $ref: '#/components/schemas/TokenLogprob'
          description: Most likely tokens and their log probabilities at this position
    TokenLogprob:
      type: object
      description: Log probability information for a single token alternative
      properties:
        token:
          type: string
          description: The text representation of the token
        logprob:
          type: number
          description: The log probability of this token
        bytes:
          type: array
          items:
            type: integer
          description: The raw byte representation of the token

````