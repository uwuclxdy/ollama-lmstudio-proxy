> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# System One

> Answer choice, yes/no, and scoring questions with a local System One model. Requires Ollama v0.35.0 or later. See the [decision guide](/capabilities/decision) for examples.


<Note>
  Currently available locally in Ollama.
</Note>

* Returns one JSON response; streaming, video, tools, and generation controls are not supported.
* Requests without images must fit within 64 KiB; requests with images must fit within 32 MiB, including base64 and JSON. The complete input must fit the loaded context window. Nimble and Tev also require two token positions for scoring. Input is never truncated.


## OpenAPI

````yaml /openapi.yaml post /v1/systemone
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
  /v1/systemone:
    post:
      summary: Answer typed questions
      description: >
        Answer choice, yes/no, and scoring questions with a local System One
        model. Requires Ollama v0.35.0 or later. See the [decision
        guide](/capabilities/decision) for examples.
      operationId: systemone
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SystemOneRequest'
            example:
              model: nimble
              state: Our checkout has returned 500 errors since 9am.
              questions:
                label:
                  type: choice
                  instructions: Which label fits this ticket?
                  criteria:
                    billing: Payments and refunds
                    bug: Software errors
                    account: Login and account access
      responses:
        '200':
          description: >-
            Answers and token usage for all questions. Example probabilities and
            confidence are rounded to four decimal places; results and usage can
            vary with the model and server configuration.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SystemOneResponse'
              example:
                model: nimble
                answers:
                  label:
                    type: choice
                    choice: bug
                    probabilities:
                      billing: 0.0125
                      bug: 0.9781
                      account: 0.0093
                    confidence: 0.8906
                usage:
                  input_tokens: 174
                  output_tokens: 1
        '400':
          description: >-
            Invalid request, unsupported model or runner, or a rendered prompt
            that exceeds the loaded context. Cloud models are rejected.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '404':
          description: Local model not found. Download the model before making a request.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '413':
          description: Request body exceeds 64 KiB without images or 32 MiB with images.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: request body must not exceed 64 KiB without images
        '500':
          description: Model loading, rendering, or scoring failed.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
      x-codeSamples:
        - lang: bash
          label: Choose a label
          source: |
            curl http://localhost:11434/v1/systemone \
              -H 'Content-Type: application/json' \
              -d '{
                "model": "nimble",
                "state": "Our checkout has returned 500 errors since 9am.",
                "questions": {
                  "label": {
                    "type": "choice",
                    "instructions": "Which label fits this ticket?",
                    "criteria": {
                      "billing": "Payments and refunds",
                      "bug": "Software errors",
                      "account": "Login and account access"
                    }
                  }
                }
              }'
components:
  schemas:
    SystemOneRequest:
      type: object
      required:
        - model
        - state
        - questions
      properties:
        model:
          type: string
          pattern: \S
          description: >-
            Local model trained for System One, such as nimble. Requires
            compatible GGUF weights and a scoring-capable runner; cloud and
            MLX/Safetensors models are not supported.
        state:
          $ref: '#/components/schemas/SystemOneContent'
        images:
          type: array
          description: >-
            Base64-encoded images shared by all questions, in request order.
            Requires Clef or Clef Flash with vision weights. URLs and data URLs
            are not supported.
          items:
            type: string
            format: byte
        questions:
          type: object
          description: >-
            Named questions about the shared state. Answers are not passed to
            later questions.
          minProperties: 1
          maxProperties: 64
          propertyNames:
            pattern: \S
          additionalProperties:
            oneOf:
              - $ref: '#/components/schemas/SystemOneChoiceQuestion'
              - $ref: '#/components/schemas/SystemOneNoulQuestion'
              - $ref: '#/components/schemas/SystemOneScoreQuestion'
        keep_alive:
          oneOf:
            - type: string
            - type: number
          description: >-
            How long to keep the model loaded after the request, as a duration
            string (such as 5m) or seconds. Zero unloads after the request; a
            negative value keeps it loaded. Defaults to the server's keep-alive
            setting (5m unless configured otherwise).
    SystemOneResponse:
      type: object
      required:
        - model
        - answers
        - usage
      properties:
        model:
          type: string
          description: Model name from the request.
        answers:
          type: object
          description: Answers keyed by the question names in the request.
          additionalProperties:
            oneOf:
              - $ref: '#/components/schemas/SystemOneChoiceAnswer'
              - $ref: '#/components/schemas/SystemOneNoulAnswer'
              - $ref: '#/components/schemas/SystemOneScoreAnswer'
        usage:
          type: object
          required:
            - input_tokens
            - output_tokens
          properties:
            input_tokens:
              type: integer
              minimum: 0
              description: >-
                Total evaluated input tokens, including image positions. Shared
                context is counted again when the model scores questions
                separately.
            output_tokens:
              type: integer
              minimum: 0
              description: >-
                Tokens generated internally for scoring, including prefix
                preparation and retries. May exceed the question count; not the
                length of the JSON response.
    ErrorResponse:
      type: object
      properties:
        error:
          type: string
          description: Error message describing what went wrong
    SystemOneContent:
      description: >-
        A nonempty string, or an object or array serialized as JSON text. Not
        interpreted as chat messages or multimodal input.
      oneOf:
        - type: string
          pattern: \S
        - type: object
          additionalProperties: true
        - type: array
          items: {}
    SystemOneChoiceQuestion:
      title: Choice
      type: object
      required:
        - type
        - instructions
        - criteria
      properties:
        type:
          type: string
          enum:
            - choice
        instructions:
          $ref: '#/components/schemas/SystemOneContent'
        criteria:
          type: object
          description: >-
            Option keys mapped to descriptions, or null for a bare label. Keys
            must not be blank; ties follow the model's option order. The option
            limit depends on the model.
          minProperties: 2
          maxProperties: 255
          propertyNames:
            pattern: \S
          additionalProperties:
            type:
              - string
              - 'null'
    SystemOneNoulQuestion:
      title: Noul
      type: object
      required:
        - type
        - instructions
      properties:
        type:
          type: string
          enum:
            - noul
        instructions:
          $ref: '#/components/schemas/SystemOneContent'
        criteria:
          type: object
          description: >-
            Optional descriptions for the two outcomes. Omitted entries use
            model-specific defaults.
          properties:
            'false':
              type: string
              default: 'No'
            'true':
              type: string
              default: 'Yes'
          additionalProperties: false
    SystemOneScoreQuestion:
      title: Score
      type: object
      required:
        - type
        - instructions
        - criteria
      properties:
        type:
          type: string
          enum:
            - score
        instructions:
          $ref: '#/components/schemas/SystemOneContent'
        criteria:
          type: array
          description: >-
            Descriptions ordered from the lowest score (index 0) to the highest.
            Defines a scale from 0 to the number of criteria minus 1. The
            maximum number of levels depends on the model.
          minItems: 2
          maxItems: 26
          items:
            type: string
    SystemOneChoiceAnswer:
      title: Choice
      type: object
      required:
        - type
        - choice
        - probabilities
        - confidence
      properties:
        type:
          type: string
          enum:
            - choice
        choice:
          type: string
          description: >-
            Option key with the highest probability. Ties follow the model's
            option order.
        probabilities:
          $ref: '#/components/schemas/SystemOneProbabilities'
        confidence:
          $ref: '#/components/schemas/SystemOneConfidence'
    SystemOneNoulAnswer:
      title: Noul
      type: object
      required:
        - type
        - noul
      properties:
        type:
          type: string
          enum:
            - noul
        noul:
          type: number
          minimum: 0
          maximum: 1
          description: >-
            Probability of true among the false and true candidates. This is a
            number, not a Boolean.
    SystemOneScoreAnswer:
      title: Score
      type: object
      required:
        - type
        - score
        - legend
        - probabilities
        - confidence
      properties:
        type:
          type: string
          enum:
            - score
        score:
          type: number
          minimum: 0
          maximum: 25
          description: >-
            Probability-weighted average of the zero-based criterion indices,
            from 0 to the number of criteria minus 1. Not rounded to a level or
            normalized to 0–1.
        legend:
          type: object
          description: >-
            Zero-based indices as string keys mapped to the criterion
            descriptions.
          additionalProperties:
            type: string
        probabilities:
          $ref: '#/components/schemas/SystemOneProbabilities'
          description: Probabilities keyed by zero-based criterion indices as strings.
        confidence:
          $ref: '#/components/schemas/SystemOneConfidence'
    SystemOneProbabilities:
      type: object
      description: >-
        Probabilities normalized over the supplied candidates, summing to 1
        subject to floating-point precision.
      additionalProperties:
        type: number
        minimum: 0
        maximum: 1
    SystemOneConfidence:
      type: number
      minimum: 0
      maximum: 1
      description: >-
        Distribution concentration, calculated as 1 - H(p) / ln(N), where H(p)
        is entropy and N is the candidate count. Zero means uniform
        probabilities; values near 1 mean one candidate dominates. Not
        calibrated correctness.

````