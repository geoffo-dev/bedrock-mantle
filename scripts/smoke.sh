#!/usr/bin/env bash
# Smoke-test a running gateway against real Bedrock Mantle.
#
#   GATEWAY=http://localhost:8080 API_KEY=... ./scripts/smoke.sh
#   CLAUDE_MODEL=claude-opus-5-5 GPT_MODEL=gpt-5.5 EMBED_MODEL=titan-embed-v2 ./scripts/smoke.sh
set -euo pipefail

GATEWAY="${GATEWAY:-http://localhost:8080}"
API_KEY="${API_KEY:?set API_KEY to the gateway key}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-5-5}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5}"
OSS_MODEL="${OSS_MODEL:-openai.gpt-oss-120b}"
EMBED_MODEL="${EMBED_MODEL:-titan-embed-v2}"
AUTH=(-H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json")
fail=0

step() { printf '\n== %s\n' "$*"; }
check() {  # check <name> <cmd...>: run, report pass/fail, keep going
  local name=$1; shift
  if out=$("$@" 2>&1); then echo "PASS  ${name}"; echo "${out}" | head -c 600; echo
  else echo "FAIL  ${name}"; echo "${out}" | head -c 1200; echo; fail=1; fi
}

chat() {  # chat <model> <stream:true|false>
  curl -sS --fail-with-body -N "${GATEWAY}/v1/chat/completions" "${AUTH[@]}" \
    -d "{\"model\":\"$1\",\"stream\":$2,\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}]}"
}

step "health";            check health curl -sS --fail-with-body "${GATEWAY}/health"
step "models";            check models curl -sS --fail-with-body "${GATEWAY}/v1/models" "${AUTH[@]}"
step "Claude (stream)";   check claude-stream chat "${CLAUDE_MODEL}" true
step "Claude (json)";     check claude-json chat "${CLAUDE_MODEL}" false
step "GPT (stream)";      check gpt-stream chat "${GPT_MODEL}" true
step "open-weight";       check oss chat "${OSS_MODEL}" false
step "embeddings";        check embeddings curl -sS --fail-with-body "${GATEWAY}/v1/embeddings" "${AUTH[@]}" \
                            -d "{\"model\":\"${EMBED_MODEL}\",\"input\":[\"hello\",\"world\"]}"
step "Anthropic passthrough"
check messages curl -sS --fail-with-body "${GATEWAY}/v1/messages" -H "x-api-key: ${API_KEY}" \
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d "{\"model\":\"${CLAUDE_MODEL}\",\"max_tokens\":100,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"

echo
if [ "$fail" -eq 0 ]; then echo "All smoke checks passed"; else echo "Some smoke checks failed"; exit 1; fi
