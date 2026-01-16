#!/bin/bash
set -e

echo "🚀 Starting MediaNet update..."

# Pull latest code
echo "📥 Pulling latest code..."
git reset --hard origin/main
git pull origin main

# Create necessary directories
mkdir -p data feeds logs config

# Build and restart containers
echo "🔨 Rebuilding Docker images..."
docker-compose build --no-cache


# Cleanup old images
echo "🧹 Cleaning up old images..."
docker image prune -f

# Show logs
echo "📋 Container status:"
docker-compose ps

echo ""
echo "🔄 To restart te container:"
echo "docker-compose down"
echo "docker-compose up -d"
echo ""
echo "📝 To view logs in real-time:"
echo "   docker-compose logs -f mon-app"
echo ""
echo "📝 To view logs from a specific time:"
echo "   docker-compose logs --tail=100 mon-app"
echo ""
echo "🛑 To stop the application:"
echo "   docker-compose down"
echo ""
echo "✅ Update completed successfully!"