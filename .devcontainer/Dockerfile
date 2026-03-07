FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install Claude Code dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
