#!/bin/bash
set -e

git reset --hard origin/main
git pull origin main
docker-compose build
docker-compose up -d
docker image prune -f

echo "✅ Mise à jour terminée"