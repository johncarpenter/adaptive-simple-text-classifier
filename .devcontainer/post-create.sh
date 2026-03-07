#!/bin/bash
set -e

echo "==> Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

echo "==> Claude Code installed successfully"
claude --version
