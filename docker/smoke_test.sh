#!/bin/bash
# Offline smoke test for a built diart-server image.
#
# Runs the container with --network none (proving no runtime network
# access is needed) and probes /readyz + /v1/info from inside the
# container. Probes use python's stdlib (always present) instead of
# curl/jq so the runtime image stays slim.
#
# Usage: bash docker/smoke_test.sh <image-tag>

set -e
IMAGE=${1:?"usage: smoke_test.sh <image-tag>"}
NAME=diart-smoke
TIMEOUT=120  # seconds; CPU warm-up of segmentation+embedding takes 30-60s

docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME --network none $IMAGE
trap "docker rm -f $NAME >/dev/null 2>&1 || true" EXIT

probe() {
    docker exec $NAME python -c "
import sys, urllib.request
try:
    body = urllib.request.urlopen('http://localhost:7007$1', timeout=2).read().decode()
    sys.stdout.write(body)
except Exception as exc:
    sys.exit(1)
"
}

echo "[smoke] waiting up to ${TIMEOUT}s for /readyz..."
for ((i=1; i<=TIMEOUT/2; i++)); do
    if probe /readyz >/dev/null 2>&1; then
        echo "[smoke] /readyz ready after $((i*2))s"
        break
    fi
    sleep 2
done

# Final probe; if still not ready, dump logs and fail.
if ! probe /readyz >/dev/null 2>&1; then
    echo "[smoke] readyz timeout — container logs follow:" >&2
    docker logs $NAME 2>&1 | tail -30 >&2
    exit 1
fi

# Verify all 10 models are served.
docker exec $NAME python -c "
import json, sys, urllib.request
body = urllib.request.urlopen('http://localhost:7007/v1/info', timeout=2).read()
info = json.loads(body)
n = len(info.get('models', {}))
if n < 10:
    print(f'expected >= 10 models, got {n}', file=sys.stderr)
    sys.exit(1)
print(f'models={n} device={info.get(\"device\")} version={info.get(\"version\")}')
"

echo "[smoke] OK"
