#!/usr/bin/env bash
# Build and (re)start the Flask backend container on port 7860.
set -e
cd "$(dirname "$0")"
docker build -t superkart-backend .
docker rm -f superkart-backend >/dev/null 2>&1 || true
docker run -d --name superkart-backend --restart unless-stopped -p 7860:7860 superkart-backend
