> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# System One

> Answer choice, yes/no, and scoring questions with a System One model, locally or in Ollama's cloud. Requires Ollama v0.35.0 or later, or v0.40.5 or later for cloud models. See the [decision guide](/capabilities/decision) for examples.


* Returns one JSON response; streaming, video, tools, and generation controls are not supported.
* Local requests' text and schema, excluding images, must fit within 64 KiB, and the whole request within 32 MiB, including base64 and JSON. The complete input must fit the loaded context window. Nimble and Tev also require two token positions for scoring. Input is never truncated.
* Cloud models need a paid plan or purchased [usage credits](https://ollama.com/pricing). Cloud requests accept up to 64 questions, named with up to 100 ASCII letters, digits, `_`, `.`, or `-`, and up to 4 images of at most 4 MiB and 16 megapixels each, 8 MiB in total. Requests must fit within 13 MiB, and input must fit the model's 65,536-token context window.


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
        Answer choice, yes/no, and scoring questions with a System One model,
        locally or in Ollama's cloud. Requires Ollama v0.35.0 or later, or
        v0.40.5 or later for cloud models. See the [decision
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
            that exceeds the loaded context or a cloud model's token limit.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '401':
          description: >-
            A cloud model was requested without signing in, or with an invalid
            API key. Through a local server, the response includes a
            `signin_url`.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '402':
          description: >-
            The cloud model isn't included in the Free plan. Add usage credits
            or upgrade. Also returned when a payment fails.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '404':
          description: Model not found. Download a local model before making a request.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '413':
          description: >-
            Local text and schema exceed 64 KiB or the request exceeds 32 MiB,
            or a cloud request exceeds 13 MiB or the model's context window.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: text and schema must not exceed 64 KiB
        '500':
          description: Model loading, rendering, or scoring failed.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
      security:
        - {}
        - bearerAuth: []
      servers:
        - url: http://localhost:11434
          description: Ollama
        - url: https://ollama.com
          description: Ollama's cloud
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
            Model trained for System One, such as nimble, or clef:cloud for
            Ollama's cloud.
        state:
          $ref: '#/components/schemas/SystemOneContent'
        images:
          type: array
          description: >-
            Base64-encoded images shared by all questions, in request order.
            Requires a model with vision, such as Clef. Models with audio input
            also accept WAV clips here. URLs and data URLs are not supported.
          items:
            type: string
            format: byte
        questions:
          type: object
          description: >-
            Named questions about the shared state. Answers are not passed to
            later questions. Cloud models accept names of up to 100 ASCII
            letters, digits, `_`, `.`, or `-`.
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
            'true':
              type: string
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
            Defines a scale from 0 to the number of criteria minus 1.
          minItems: 2
          maxItems: 10
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
          maximum: 9
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
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: API Key

````