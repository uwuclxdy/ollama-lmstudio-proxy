> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Cloud usage

> Usage statistics for cloud inference, web search, and web fetch.

<RequestExample>
  ```shell Request theme={"system"}
  curl 'https://ollama.com/api/usage?range=24h' \
    -H "Authorization: Bearer $OLLAMA_API_KEY"
  ```
</RequestExample>

<ResponseExample>
  ```json Response (22 empty buckets omitted) theme={"system"}
  {
    "range": "24h",
    "scope": "self",
    "granularity": "hour",
    "from": "2026-09-30T02:00:00Z",
    "until": "2026-10-01T02:30:00Z",
    "totals": {
      "request_count": 15,
      "usage_usd": 0.01718,
      "input_tokens": 106000,
      "cached_input_tokens": 46000,
      "output_tokens": 13600
    },
    "buckets": [
      {
        "from": "2026-09-30T12:00:00Z",
        "until": "2026-09-30T13:00:00Z",
        "request_count": 4,
        "usage_usd": 0.0052,
        "input_tokens": 24000,
        "cached_input_tokens": 0,
        "output_tokens": 3200
      },
      {
        "from": "2026-10-01T00:00:00Z",
        "until": "2026-10-01T01:00:00Z",
        "request_count": 8,
        "usage_usd": 0.0088,
        "input_tokens": 64000,
        "cached_input_tokens": 40000,
        "output_tokens": 8000
      },
      {
        "from": "2026-10-01T02:00:00Z",
        "until": "2026-10-01T02:30:00Z",
        "partial": true,
        "request_count": 3,
        "usage_usd": 0.00318,
        "input_tokens": 18000,
        "cached_input_tokens": 6000,
        "output_tokens": 2400
      }
    ]
  }
  ```
</ResponseExample>

New usage may take time to appear.

For your remaining included and purchased credits, use [Balance](/api/balance).

Unsupported or repeated query parameters return `400`. Errors follow Ollama's [JSON error format](/api/errors#error-messages).

## Time ranges

| Range | Starts at | Bucket size | Buckets |
| - | - | - | - |
| `24h` | 24 hours before the start of the current UTC hour | Hour | 25 |
| `7d` | Seven days ago at 00:00 UTC | Day | 8 |
| `30d` | Thirty days ago at 00:00 UTC | Day | 31 |

Every range ends at the time of the request. For example, at 02:30 UTC on October 1, `24h` starts at 02:00 UTC on September 30. It returns 24 complete hourly buckets followed by the current 02:00–02:30 bucket. The `7d` and `30d` ranges return seven and thirty complete days, followed by today so far.

Buckets appear in chronological order, including zeroes for hours or days with no recorded requests. The final bucket covers the current hour or day and has `partial: true`.

All timestamps are in UTC. Each `from` is inclusive and each `until` is exclusive.

## Usage totals and buckets

`totals` summarizes usage across the entire range. `buckets` breaks that usage into hours or days.

`granularity` is set automatically: `hour` for `24h` and `day` for `7d` and `30d`. The top-level `from` and `until` describe the entire range; each bucket has its own boundaries.

`partial` indicates whether the hour or day is still in progress, rather than whether all usage has been recorded.

<Accordion title="Legacy plans">
  For legacy plans, only request counts are available. If a bucket or total includes legacy requests, cost and token counts are omitted.
</Accordion>

## Rate limits

This endpoint allows 10 requests per minute per user, shared across API keys and devices. We recommend polling once per minute. A `429` response includes a `Retry-After` header with the number of seconds to wait before retrying.

## Team usage

By default, everyone sees their own usage (`scope=self`).

Team admins can use `scope=team` to see usage for the whole team:

```shell theme={"system"}
curl 'https://ollama.com/api/usage?range=7d&scope=team' \
  -H "Authorization: Bearer $OLLAMA_API_KEY"
```

## Coming soon

We plan to add custom time ranges, usage breakdowns by model and API key, and breakdowns by team member for team admins.

## Reference


## OpenAPI

````yaml openapi.yaml GET /api/usage
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
  /api/usage:
    get:
      summary: Cloud usage
      description: Usage statistics for cloud inference, web search, and web fetch.
      operationId: usage
      parameters:
        - name: range
          in: query
          description: >-
            Completed UTC hours or days to include, plus the current partial
            hour or day.
          schema:
            type: string
            enum:
              - 24h
              - 7d
              - 30d
            default: 7d
        - name: scope
          in: query
          description: >-
            Your requests, or your team's requests. Team scope requires being a
            team admin.
          schema:
            type: string
            enum:
              - self
              - team
            default: self
      responses:
        '200':
          description: Usage for the selected range and scope.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UsageResponse'
        '400':
          description: Invalid query parameters.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '401':
          description: Invalid credentials.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '403':
          description: Account suspended or team scope requested without team admin access.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '429':
          description: Too many requests.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
          headers:
            Retry-After:
              description: Seconds to wait before retrying.
              schema:
                type: integer
                minimum: 1
        '503':
          description: Usage reporting is temporarily unavailable.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
      security:
        - bearerAuth: []
      servers:
        - url: https://ollama.com
          description: Ollama Cloud
components:
  schemas:
    UsageResponse:
      type: object
      required:
        - range
        - scope
        - granularity
        - from
        - until
        - totals
        - buckets
      properties:
        range:
          type: string
          enum:
            - 24h
            - 7d
            - 30d
          description: Selected time range.
        scope:
          type: string
          enum:
            - self
            - team
          description: Selected usage scope.
        granularity:
          type: string
          enum:
            - hour
            - day
          description: 'Set automatically: hour for 24h, day for 7d and 30d.'
        from:
          type: string
          format: date-time
          description: Start of the range in UTC, inclusive.
        until:
          type: string
          format: date-time
          description: End of the range in UTC, exclusive.
        totals:
          $ref: '#/components/schemas/UsageMetrics'
          description: Usage across the entire range.
        buckets:
          type: array
          description: >-
            Usage in chronological order, including buckets with no recorded
            requests.
          items:
            $ref: '#/components/schemas/UsageBucket'
    ErrorResponse:
      type: object
      properties:
        error:
          type: string
          description: Error message describing what went wrong
    UsageMetrics:
      type: object
      required:
        - request_count
      properties:
        request_count:
          type: integer
          minimum: 0
          description: Number of recorded requests.
        usage_usd:
          type: number
          description: >-
            Total value of the requests in USD, including usage covered by your
            plan and usage paid from purchased credits.
        input_tokens:
          type: integer
          description: Recorded input tokens, including cached input tokens.
        cached_input_tokens:
          type: integer
          description: >-
            Recorded input tokens read from cache, already included in
            input_tokens.
        output_tokens:
          type: integer
          description: Recorded output tokens.
    UsageBucket:
      allOf:
        - $ref: '#/components/schemas/UsageMetrics'
        - type: object
          required:
            - from
            - until
          properties:
            from:
              type: string
              format: date-time
              description: Start of the bucket in UTC, inclusive.
            until:
              type: string
              format: date-time
              description: End of the bucket in UTC, exclusive.
            partial:
              type: boolean
              description: >-
                Present and true for the current hour or day, which is still in
                progress.
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: API Key

````