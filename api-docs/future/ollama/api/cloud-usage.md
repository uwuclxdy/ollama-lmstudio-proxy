> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Cloud usage

> Usage statistics for cloud inference, web search, and web fetch.

```shell Request theme={"system"}
curl 'https://ollama.com/api/usage?range=24h' \
  -H "Authorization: Bearer $OLLAMA_API_KEY"
```

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

New usage may take time to appear.

For your remaining included and purchased credits, use [Balance](https://docs.ollama.com/api/balance.md).

Unsupported or repeated query parameters return `400`. Errors follow Ollama's [JSON error format](https://docs.ollama.com/api/errors.md#error-messages).

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

## Model breakdown

Add `group_by=model` to see usage by model:

```shell theme={"system"}
curl 'https://ollama.com/api/usage?range=7d&group_by=model' \
  -H "Authorization: Bearer $OLLAMA_API_KEY"
```

The response includes `groups` in both `totals` and every bucket.

```json Response excerpt theme={"system"}
{
  "totals": {
    "groups": [
      {
        "model": "glm-5.3-flash",
        "request_count": 12,
        "usage_usd": 0.0132
      }
    ]
  },
  "buckets": [
    {
      "from": "2026-09-30T00:00:00Z",
      "until": "2026-10-01T00:00:00Z",
      "groups": [
        {
          "model": "glm-5.3-flash",
          "request_count": 6,
          "usage_usd": 0.0066
        }
      ]
    }
  ]
}
```

> [accordion] Example with multiple models and buckets
>
> ```json Response (6 empty buckets omitted) theme={"system"}
> {
>   "range": "7d",
>   "scope": "self",
>   "granularity": "day",
>   "group_by": "model",
>   "from": "2026-09-24T00:00:00Z",
>   "until": "2026-10-01T02:30:00Z",
>   "totals": {
>     "request_count": 16,
>     "usage_usd": 0.0268,
>     "input_tokens": 116000,
>     "cached_input_tokens": 60000,
>     "output_tokens": 15000,
>     "groups": [
>       {
>         "model": "glm-5.3-flash",
>         "request_count": 12,
>         "usage_usd": 0.0132,
>         "input_tokens": 96000,
>         "cached_input_tokens": 60000,
>         "output_tokens": 12000
>       },
>       {
>         "model": "glm-5.3",
>         "request_count": 2,
>         "usage_usd": 0.01,
>         "input_tokens": 4000,
>         "cached_input_tokens": 0,
>         "output_tokens": 1000
>       },
>       {
>         "model": "gpt-oss:120b",
>         "request_count": 2,
>         "usage_usd": 0.0036,
>         "input_tokens": 16000,
>         "cached_input_tokens": 0,
>         "output_tokens": 2000
>       }
>     ]
>   },
>   "buckets": [
>     {
>       "from": "2026-09-30T00:00:00Z",
>       "until": "2026-10-01T00:00:00Z",
>       "request_count": 8,
>       "usage_usd": 0.0166,
>       "input_tokens": 52000,
>       "cached_input_tokens": 30000,
>       "output_tokens": 7000,
>       "groups": [
>         {
>           "model": "glm-5.3-flash",
>           "request_count": 6,
>           "usage_usd": 0.0066,
>           "input_tokens": 48000,
>           "cached_input_tokens": 30000,
>           "output_tokens": 6000
>         },
>         {
>           "model": "glm-5.3",
>           "request_count": 2,
>           "usage_usd": 0.01,
>           "input_tokens": 4000,
>           "cached_input_tokens": 0,
>           "output_tokens": 1000
>         }
>       ]
>     },
>     {
>       "from": "2026-10-01T00:00:00Z",
>       "until": "2026-10-01T02:30:00Z",
>       "partial": true,
>       "request_count": 8,
>       "usage_usd": 0.0102,
>       "input_tokens": 64000,
>       "cached_input_tokens": 30000,
>       "output_tokens": 8000,
>       "groups": [
>         {
>           "model": "glm-5.3-flash",
>           "request_count": 6,
>           "usage_usd": 0.0066,
>           "input_tokens": 48000,
>           "cached_input_tokens": 30000,
>           "output_tokens": 6000
>         },
>         {
>           "model": "gpt-oss:120b",
>           "request_count": 2,
>           "usage_usd": 0.0036,
>           "input_tokens": 16000,
>           "cached_input_tokens": 0,
>           "output_tokens": 2000
>         }
>       ]
>     }
>   ]
> }
> ```

Currently, up to 10 groups appear under `groups`, ranked by total `usage_usd` across the range.

Every bucket uses the same selection and order. Groups with no requests in a bucket are omitted; an empty bucket has `groups: []`.

Additional groups are combined under `remainder`, omitted when empty. Totals always include all usage.

> [accordion] Legacy plans
>
> For usage that includes requests from legacy plans, only request counts are available. If the range includes legacy requests, groups are ranked by request count instead of cost.

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

We plan to add custom time ranges, usage breakdowns by API key, and breakdowns by team member for team admins.

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
        - name: group_by
          in: query
          description: >-
            How to group usage within totals and buckets. Supported values:
            model.
          schema:
            type: string
            enum:
              - model
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
          description: Ollama's cloud
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
        group_by:
          type: string
          enum:
            - model
          description: Selected grouping. Present when group_by is requested.
        from:
          type: string
          format: date-time
          description: Start of the range in UTC, inclusive.
        until:
          type: string
          format: date-time
          description: End of the range in UTC, exclusive.
        totals:
          $ref: '#/components/schemas/UsageSummary'
          description: Usage across the entire range, including groups in the remainder.
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
    UsageSummary:
      allOf:
        - $ref: '#/components/schemas/UsageMetrics'
        - type: object
          properties:
            groups:
              type: array
              description: >-
                Present when group_by is requested. Currently returns up to 10
                groups selected across the entire range, ranked by usage_usd or
                by request_count when the range includes legacy requests. Groups
                use the same order in every bucket; groups without requests in a
                bucket are omitted.
              items:
                $ref: '#/components/schemas/UsageGroup'
            remainder:
              $ref: '#/components/schemas/UsageMetrics'
              description: >-
                Combined usage from groups outside the current limit. Omitted
                when there are no requests from additional groups.
    UsageBucket:
      allOf:
        - $ref: '#/components/schemas/UsageSummary'
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
    UsageGroup:
      allOf:
        - $ref: '#/components/schemas/UsageMetrics'
        - type: object
          required:
            - model
          properties:
            model:
              type: string
              description: Model name.
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: API Key

````