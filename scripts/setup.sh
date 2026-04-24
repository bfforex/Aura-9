#!/bin/bash
set -e

ollama pull qwen3.5:9b-instruct-q5_k_m
ollama pull qwen3.5:9b-instruct-q4_k_m
ollama pull nomic-embed-text

docker compose up -d

sleep 5

aura9 migrations run
