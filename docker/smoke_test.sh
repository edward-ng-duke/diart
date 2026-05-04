#!/bin/bash
set -e
IMAGE=$1
PORT=17007
docker run -d --name diart-smoke --network none -p $PORT:7007 $IMAGE
trap "docker rm -f diart-smoke" EXIT

for i in {1..30}; do
  if curl -sf http://localhost:$PORT/readyz; then break; fi
  sleep 2
done
curl -sf http://localhost:$PORT/readyz || { echo "readyz timeout"; exit 1; }
curl -sf http://localhost:$PORT/v1/info | jq -e '.models | length >= 10'
echo "[smoke] OK"
