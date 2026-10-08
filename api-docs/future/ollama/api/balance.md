> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Balance

> Check your remaining usage credits.

<RequestExample>
  ```shell Request theme={"system"}
  curl https://ollama.com/api/balance \
    -H "Authorization: Bearer $OLLAMA_API_KEY"
  ```
</RequestExample>

<ResponseExample>
  ```json Response theme={"system"}
  {
    "included": {
      "balance_usd": 72.5,
      "allowance_usd": 100,
      "period": {
        "from": "2026-09-15T09:30:00Z",
        "until": "2026-10-15T09:30:00Z"
      }
    },
    "purchased": {
      "balance_usd": 25
    }
  }
  ```
</ResponseExample>

This endpoint takes no query parameters. For usage history, see [Cloud usage](/api/cloud-usage).

## Balances

The included period follows your plan's monthly reset schedule, including for annual subscriptions. Period timestamps are in UTC, with `from` inclusive and `until` exclusive.

The purchased balance includes only unexpired credits.

<Accordion title="Legacy plans">
  Plans with session and weekly limits return `session` and `weekly` in `included`:

  ```json theme={"system"}
  {
    "included": {
      "session": {
        "remaining_percent": 75,
        "resets_at": "2026-10-01T07:00:00Z"
      },
      "weekly": {
        "remaining_percent": 40,
        "resets_at": "2026-10-05T00:00:00Z"
      }
    },
    "purchased": {
      "balance_usd": 25
    }
  }
  ```

  `remaining_percent` ranges from `0` to `100`; `75` means 75% remains. `resets_at` is the next reset time in UTC.
</Accordion>

## Rate limits

This endpoint allows 10 requests per minute per user, shared across API keys and devices. We recommend polling once per minute. A `429` response includes a `Retry-After` header with the number of seconds to wait before retrying.

## Teams

All team members share the same included and purchased balances.

## Reference


## OpenAPI

````yaml openapi.yaml GET /api/balance
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
  /api/balance:
    get:
      summary: Balance
      description: Check your remaining usage credits.
      operationId: balance
      responses:
        '200':
          description: Current included and purchased balances.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BalanceResponse'
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
          description: Account suspended.
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
          description: Balance reporting is temporarily unavailable.
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
    BalanceResponse:
      type: object
      required:
        - included
        - purchased
      properties:
        included:
          oneOf:
            - $ref: '#/components/schemas/IncludedBalance'
            - $ref: '#/components/schemas/LegacyIncludedBalance'
        purchased:
          type: object
          required:
            - balance_usd
          properties:
            balance_usd:
              type: number
              description: Remaining unexpired purchased credits in USD.
    ErrorResponse:
      type: object
      properties:
        error:
          type: string
          description: Error message describing what went wrong
    IncludedBalance:
      title: Included credits
      type: object
      required:
        - balance_usd
        - allowance_usd
        - period
      properties:
        balance_usd:
          type: number
          description: Remaining included credits in USD.
        allowance_usd:
          type: number
          description: Included credits available for the full period in USD.
        period:
          type: object
          required:
            - from
            - until
          properties:
            from:
              type: string
              format: date-time
              description: Start of the included period in UTC, inclusive.
            until:
              type: string
              format: date-time
              description: End of the included period in UTC, exclusive.
    LegacyIncludedBalance:
      title: Legacy plan limits
      type: object
      required:
        - session
        - weekly
      properties:
        session:
          $ref: '#/components/schemas/LegacyBalanceLimit'
        weekly:
          $ref: '#/components/schemas/LegacyBalanceLimit'
    LegacyBalanceLimit:
      type: object
      required:
        - remaining_percent
        - resets_at
      properties:
        remaining_percent:
          type: number
          minimum: 0
          maximum: 100
          description: Percentage of the limit remaining.
        resets_at:
          type: string
          format: date-time
          description: Next reset time in UTC.
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: API Key

````