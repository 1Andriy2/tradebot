#!/usr/bin/env bash
# Run a command with this project's .env loaded, without touching global config.
#   ./scripts/with-env.sh gh repo view
#   ./scripts/with-env.sh claude          # start Claude Code so .mcp.json can expand tokens
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "with-env: $ENV_FILE not found (copy .env.example to .env)" >&2
  exit 1
fi

# Never let the project env change Claude Code's own account/auth.
if grep -Eq '^[[:space:]]*(export[[:space:]]+)?(ANTHROPIC_|CLAUDE_)' "$ENV_FILE"; then
  echo "with-env: ANTHROPIC_*/CLAUDE_* variables are not allowed in .env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Keep CLI tools from reading/writing global config dirs where possible.
export GH_CONFIG_DIR="$ROOT/.local/gh"
mkdir -p "$GH_CONFIG_DIR"

exec "$@"
