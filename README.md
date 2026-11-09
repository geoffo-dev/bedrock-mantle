# Mantle Gateway

A small, self-hosted **OpenAI-compatible gateway for Amazon Bedrock Mantle**. Point Open WebUI (or any OpenAI client) at it to use **Claude**, **GPT-5.x / GPT-6.x**, and the open-weight models (gpt-oss, Qwen, DeepSeek, Kimi, Gemma, Grok) on Bedrock. It also handles **embeddings** for RAG. It is a drop-in replacement for the retired Bedrock Access Gateway: one container, free, with no AWS-side infrastructure.

## Why it's needed

Bedrock Mantle (`https://bedrock-mantle.<region>.api.aws`) serves three different API surfaces from one host. Open WebUI only speaks Chat Completions against one base URL:

| Models | Mantle surface | Works directly in Open WebUI? |
|---|---|---|
| gpt-oss, Qwen, DeepSeek, Kimi, … | `/v1/chat/completions` | yes |
| `openai.gpt-5.x`, `gpt-6.x` | `/openai/v1/responses` (some are Responses-only) | **no** |
| Gemma 4, Grok | `/openai/v1/chat/completions` | **no** (different base) |
| Claude (`anthropic.claude-*`) | `/anthropic/v1/messages` (Anthropic Messages API) | **no** |

The gateway exposes one OpenAI-compatible `/v1` and routes each model to the right surface. It translates requests and **streaming** responses both ways, including images, tool calling, reasoning ("thinking") display and usage.

## Endpoints

| Gateway endpoint | Goes to |
|---|---|
| `GET /v1/models` | Mantle `/v1/models` merged with `models.yaml` (aliases, names, hide / allow-list) |
| `POST /v1/chat/completions` | Claude → Anthropic Messages · GPT-5.x → Responses · others → Chat passthrough |
| `POST /v1/embeddings` | bedrock-runtime `InvokeModel` (Titan Text Embeddings v1/v2, Cohere Embed v3/v4) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Mantle `/anthropic/v1/messages…` passthrough (for Claude Code and other Anthropic SDK clients) |
| `POST /v1/responses` | Mantle `/openai/v1/responses` or `/v1/responses` passthrough (for Codex CLI and other Responses clients) |
| `GET /health` | liveness / readiness (no auth) |

## Quick start (Docker Compose + Open WebUI)

```bash
cp .env.example .env                      # set API_KEY, AWS_REGION and AWS credentials
docker compose up -d --build
open http://localhost:3000                # Open WebUI, already wired to the gateway
```

Open WebUI is preconfigured through environment variables. The chat connection uses `OPENAI_API_BASE_URL=http://mantle-gateway:8080/v1`. RAG embeddings use `RAG_EMBEDDING_ENGINE=openai` with `RAG_EMBEDDING_MODEL=titan-embed-v2`. To connect an existing Open WebUI instead, go to **Admin Settings → Connections → OpenAI API → +**. Use URL `http://<gateway>:8080/v1` and your `API_KEY` as the key.

**No AWS account handy?** `docker compose -f docker-compose.mock.yml up -d --build` runs the gateway against a fake Mantle (`tests/mock_mantle.py`). Add `--profile webui` to include Open WebUI.

Smoke-test a running gateway against real Bedrock:

```bash
GATEWAY=http://localhost:8080 API_KEY=... ./scripts/smoke.sh
```

## AWS credentials

Pick one:

1. **Bedrock API key**: set `AWS_BEARER_TOKEN_BEDROCK` to a Bedrock API key from the Bedrock console.
2. **IAM credentials**: leave `AWS_BEARER_TOKEN_BEDROCK` unset. The gateway uses the standard AWS credential chain (env keys, `AWS_PROFILE`, SSO, IRSA, EKS Pod Identity, ECS/EC2 roles). It mints short-lived Bedrock bearer tokens with AWS's [`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/). Tokens are re-minted every 10 minutes and on any 401/403.

IAM permissions: the principal needs `bedrock:CallWithBearerToken` plus permission to invoke the models you use, on Mantle and on bedrock-runtime for embeddings. Check the current Bedrock Mantle IAM documentation for the exact action names and any model-access enablement in your account.

Mantle is regional. Set `AWS_REGION` to a Mantle region (for example `us-east-1`, `us-east-2`, `us-west-2`, `eu-west-1`, `eu-west-2`, `eu-central-1`, `eu-north-1`, `eu-south-1`, `ap-northeast-1`, `ap-south-1`, `ap-southeast-3`, `sa-east-1`). Model availability differs by region.

## Configuration

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | – (required) | Key(s) clients must send, comma-separated. Accepted as `Authorization: Bearer` or `x-api-key`. |
| `ALLOW_NO_AUTH` | `false` | Allow requests without a key (only for isolated networks). |
| `AWS_REGION` | `us-east-1` | Mantle / bedrock-runtime region. |
| `AWS_BEARER_TOKEN_BEDROCK` | – | Bedrock API key. If unset, tokens are minted from IAM credentials. |
| `MANTLE_BASE_URL` / `BEDROCK_RUNTIME_URL` | regional defaults | Endpoint overrides (VPC endpoints, testing). |
| `MODELS_CONFIG` | `/app/config/models.yaml` | Path to the model catalog. |
| `MODEL_DISCOVERY` | `true` | Merge Mantle's `/v1/models` into the list. |
| `MODEL_CACHE_TTL` | `600` | Seconds to cache discovered models. |
| `REASONING_FORMAT` | `reasoning_content` | How thinking is returned: `reasoning_content` (Open WebUI shows a "Thinking" panel), `think_tags` (`<think>…</think>` in the content), or `none`. |
| `ANTHROPIC_VERSION` | `2023-06-01` | `anthropic-version` sent to the Claude surface. |
| `CLAUDE_PROMPT_CACHING` | `false` | Add top-level `cache_control` (automatic prompt caching) to Claude requests. |
| `FETCH_REMOTE_IMAGES` | `true` | Inline http(s) image URLs for Claude. Private / internal addresses are always refused. |
| `REQUEST_TIMEOUT` | `600` | Upstream timeout in seconds. |
| `LOG_LEVEL` | `INFO` | Prompts and completions are never logged. |

### `models.yaml`

See [`config/models.example.yaml`](config/models.example.yaml). Every entry is optional. Without a file, the gateway lists whatever Mantle returns and applies the routing rules below.

```yaml
discovery: true          # also list models discovered from Mantle
allow_list: false        # true = expose only the models below
hide: ["*safeguard*"]    # hide discovered models by glob
models:
  - id: anthropic.claude-opus-5-5   # Bedrock model ID
    alias: claude-opus-5-5          # ID clients use / see
    name: Claude Opus 5.5           # display name in Open WebUI
    default_effort: high
  - id: amazon.titan-embed-text-v2:0
    alias: titan-embed-v2
    type: embedding                 # hidden from /v1/models, used by /v1/embeddings
    dimensions: 1024
```

Routing rules (override with `api:` / `base:` per model):

| Model ID | `api` | `base` |
|---|---|---|
| contains `claude` | `messages` | – |
| `openai.gpt-<number>…` | `responses` | `openai/v1` |
| `google.gemma-4*`, `xai.grok*` | `chat` | `openai/v1` |
| anything else | `chat` | `v1` |

Anthropic-style IDs such as `claude-opus-5-5` are mapped to `anthropic.claude-opus-5-5` automatically.

**`api: bedrock`** sends Claude to bedrock-runtime `InvokeModel` (same Messages translation, streamed over AWS event stream) instead of Mantle. Use it where Mantle's regional catalogue lacks a Claude model, or for `eu.` / `global.` inference profiles, which Mantle doesn't support. Mantle eu-west-2, for example, serves no Claude, while bedrock-runtime eu-west-2 has `anthropic.claude-opus-5-5` and `eu.anthropic.claude-sonnet-5-5`. Set the model's `id` to the Bedrock model or inference-profile ID. IAM needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the model or profile. The `/v1/messages` passthrough still goes to Mantle.

## How requests are translated

**Claude (Chat Completions → Anthropic Messages)**

- Message content:
  - `system` / `developer` messages → `system`.
  - Images: data-URI images become base64 `image` blocks. PDFs become `document` blocks.
  - Tool calls: `tool_calls` and `tool` messages become `tool_use` / `tool_result` blocks, with pairing repaired when a client leaves gaps.
- Request parameters:
  - `tool_choice` and `parallel_tool_calls` are mapped. Forced tool choice (`required` or a named tool) is downgraded to `auto` while thinking is on, which Claude requires, and on models that reject it outright (Opus/Sonnet 5.5+, Fable/Mythos 5.1+). Set `thinking: none` on a model to keep forced tool choice.
  - `response_format: json_schema` → structured outputs (`output_config.format`).
  - `stop` → `stop_sequences`.
  - `max_tokens` defaults to 64k when streaming and 16k otherwise.
- **Thinking.** Current models (4.6+, 5.x) get adaptive thinking with summarized display, so Open WebUI shows the reasoning. Open WebUI's `reasoning_effort` (`low` … `max`) maps to `output_config.effort`. Older models (Haiku 4.5, Claude 4.5-era and earlier) get budget-based thinking, but only when an effort is requested.
- **Sampling.** `temperature` / `top_p` / `top_k` are dropped for current Claude models, which reject them. Enable with `allow_sampling: true` per model.
- **Earlier thinking.** Thinking from earlier turns is not replayed. Open WebUI doesn't keep thinking signatures, and leaving thinking out of the history is always valid.

**GPT-5.x / 6.x (Chat Completions → Responses)**

- system → `instructions`. Messages, images, tool calls and tool results → Responses input items.
- `max_tokens` → `max_output_tokens`. `reasoning_effort` → `reasoning.effort`, with `summary: auto` so reasoning summaries stream into Open WebUI's thinking panel.
- `response_format` → `text.format`. Sampling parameters are dropped.

**Everything else** is forwarded to Mantle's Chat Completions as-is, with the model ID resolved. The response stream is relayed byte-for-byte.

Upstream errors come back with their HTTP status in OpenAI's error shape (`{"error": {"message", "type", "code"}}`).

## Kubernetes

```bash
kubectl create namespace mantle-gateway
kubectl -n mantle-gateway create secret generic mantle-gateway \
  --from-literal=API_KEY="$(openssl rand -hex 32)" \
  --from-literal=AWS_BEARER_TOKEN_BEDROCK="<bedrock api key>"     # omit when using IRSA / Pod Identity
kubectl apply -k deploy/k8s
```

- [`deploy/k8s`](deploy/k8s) is a kustomization containing:
  - Namespace and ServiceAccount (IRSA / Pod Identity notes inside).
  - Deployment: 2 replicas, non-root, read-only root filesystem, probes on `/health`.
  - ClusterIP Service on port 8080.
  - ConfigMaps generated from `deploy/k8s/models.yaml` and a few env vars.
- Set `AWS_REGION` and the image tag in `kustomization.yaml`.
- Point an in-cluster Open WebUI at `http://mantle-gateway.mantle-gateway.svc:8080/v1` with the same `API_KEY`.

The gateway is stateless, so scale replicas freely. Streams are long-lived HTTP responses. If you put an Ingress in front, raise its read timeout and disable response buffering.

Images are built by CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) and pushed to `ghcr.io/geoffo-dev/bedrock-mantle` (amd64 + arm64) on every push to `main` and on `v*` tags.

## Other clients

- **Claude Code**: `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=$API_KEY ANTHROPIC_MODEL=claude-opus-5-5 claude`. Requests go through `/v1/messages` unchanged, so any `anthropic-beta` flags the client sends must be supported by Mantle.
- **Codex CLI / Responses clients**: base URL `http://localhost:8080/v1`, API key `$API_KEY`, wire API `responses`.
- **OpenAI SDKs / LangChain / LibreChat …**: base URL `http://localhost:8080/v1`.

## Development

```bash
uv sync
uv run pytest -q          # unit + end-to-end tests against the mock Mantle
uv run ruff check . && uv run ruff format --check .
uv run uvicorn --factory mantle_gateway.main:create_app --reload --port 8080
```

Layout: `src/mantle_gateway/`
- `routes/`: HTTP endpoints.
- `translate/`: the protocol translators.
- `catalog.py`: model discovery and routing rules.
- `auth.py`: inbound keys and Bedrock tokens.
- `upstream.py`: HTTP client, retries, error mapping.

## Limitations

- Model IDs and per-model quirks change often. Defaults are pattern-based. Override per model in `models.yaml` (`api`, `base`, `thinking`, `allow_sampling`, `forced_tool_choice`, `max_tokens`).
- Only Bearer-token auth is used upstream. IAM credentials are turned into Bearer tokens; direct SigV4 request signing isn't implemented.
- `n > 1`, `logprobs`, and audio input/output aren't supported on the Claude and Responses routes.
- Embeddings support Titan and Cohere models. Open WebUI can't say whether text is a query or a document, so Cohere `input_type` defaults to `search_document`. Override it per model, or send `input_type` in the request.
