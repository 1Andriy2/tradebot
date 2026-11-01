#!/usr/bin/env bash
# Verify every configured token belongs to the project account. Prints no secrets.
# Usage: ./scripts/with-env.sh ./scripts/check-accounts.sh
set -uo pipefail

EXPECTED="andrijkostelnij72@gmail.com"
status=0

ok()   { echo "  OK   $1"; }
fail() { echo "  FAIL $1"; status=1; }
skip() { echo "  SKIP $1"; }

echo "Expected account: $EXPECTED"

echo "[git]"
email="$(git config --local user.email || true)"
[[ "$email" == "$EXPECTED" ]] && ok "local git identity" || fail "local git email is '$email'"

echo "[GitHub]"
if [[ -n "${GH_TOKEN:-}" ]]; then
  # Requires the fine-grained PAT permission "Email addresses: read".
  emails="$(curl -fsS -H "Authorization: Bearer $GH_TOKEN" https://api.github.com/user/emails 2>/dev/null \
    | python3 -c 'import json,sys; print(" ".join(e["email"] for e in json.load(sys.stdin)))' 2>/dev/null || true)"
  [[ " $emails " == *" $EXPECTED "* ]] && ok "GH_TOKEN belongs to $EXPECTED" \
    || fail "GH_TOKEN email check failed (wrong account or missing 'Email addresses: read')"
else skip "GH_TOKEN not set"; fi

echo "[Supabase]"
if [[ -n "${SUPABASE_ACCESS_TOKEN:-}" && -n "${SUPABASE_PROJECT_REF:-}" ]]; then
  # The Management API does not expose the account email; check the token can see our project.
  code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
    "https://api.supabase.com/v1/projects/$SUPABASE_PROJECT_REF")"
  [[ "$code" == "200" ]] && ok "token has access to project $SUPABASE_PROJECT_REF (confirm owner email in dashboard)" \
    || fail "token cannot access project (HTTP $code)"
else skip "SUPABASE_ACCESS_TOKEN / SUPABASE_PROJECT_REF not set"; fi

echo "[Vercel]"
if [[ -n "${VERCEL_TOKEN:-}" ]]; then
  vemail="$(curl -fsS -H "Authorization: Bearer $VERCEL_TOKEN" https://api.vercel.com/v2/user 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["user"]["email"])' 2>/dev/null || true)"
  [[ "$vemail" == "$EXPECTED" ]] && ok "VERCEL_TOKEN belongs to $EXPECTED" || fail "VERCEL_TOKEN email check failed"
else skip "VERCEL_TOKEN not set"; fi

exit $status
