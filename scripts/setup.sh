#!/bin/bash
set -e

# Create required directory structure
mkdir -p backups/qdrant backups/falkordb logs archive secrets

# Ensure .gitkeep files exist (for git tracking)
for dir in backups/qdrant backups/falkordb logs archive secrets; do
    touch "${dir}/.gitkeep"
done

ollama pull qwen3.5:9b-instruct-q5_k_m
ollama pull qwen3.5:9b-instruct-q4_k_m
ollama pull nomic-embed-text

docker compose up -d

sleep 5

aura9 migrations run
