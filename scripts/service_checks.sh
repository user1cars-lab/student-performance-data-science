#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
uvicorn src.api:app --host 127.0.0.1 --port 8765 >/tmp/student-api.log 2>&1 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true; kill "$STREAMLIT_PID" 2>/dev/null || true' EXIT
for _ in {1..20}; do curl -fsS http://127.0.0.1:8765/health >/tmp/health.json && break || sleep 0.5; done
grep -q '"status":"ok"' /tmp/health.json
curl -fsS -X POST http://127.0.0.1:8765/predict -H 'content-type: application/json' -d '{}' >/tmp/predict.json
grep -Eq 'predicted_grade|performance_level' /tmp/predict.json
streamlit run app/streamlit_app.py --server.headless true --server.port 8766 >/tmp/student-streamlit.log 2>&1 &
STREAMLIT_PID=$!
for _ in {1..20}; do curl -fsS http://127.0.0.1:8766/_stcore/health >/tmp/streamlit-health.json && break || sleep 0.5; done
grep -q 'ok' /tmp/streamlit-health.json
printf 'API health: '; cat /tmp/health.json; printf '\nAPI prediction: '; cat /tmp/predict.json; printf '\nStreamlit health: '; cat /tmp/streamlit-health.json
