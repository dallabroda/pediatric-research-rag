#!/bin/bash
# ============================================================
# setup_claude_code.sh
# Run this ONCE to configure Claude Code for this project.
# Prerequisites: Claude Code installed, AWS CLI configured.
# ============================================================

set -e

echo "🔧 Setting up Claude Code for pediatric-research-rag..."
echo ""

# ----------------------------------------------------------
# 1. Install Claude Code Skills (plugins)
# ----------------------------------------------------------
echo "📦 Installing AWS Skills plugin..."
claude plugin install aws-common@aws-skills 2>/dev/null || echo "  ⚠️  aws-common already installed or install manually"
claude plugin install serverless-eda@aws-skills 2>/dev/null || echo "  ⚠️  serverless-eda already installed or install manually"

echo ""

# ----------------------------------------------------------
# 2. Add MCP Servers to project
# ----------------------------------------------------------
echo "🔌 Adding MCP servers..."

# AWS Documentation — look up Bedrock, Lambda, S3 APIs
claude mcp add awslabs.aws-documentation-mcp-server -s project \
  -e FASTMCP_LOG_LEVEL=WARNING \
  -- uvx awslabs.aws-documentation-mcp-server@latest

# S3 — manage buckets and objects
claude mcp add awslabs.s3-mcp-server -s project \
  -e AWS_PROFILE="${AWS_PROFILE:-default}" \
  -e AWS_REGION="${AWS_REGION:-us-east-1}" \
  -e FASTMCP_LOG_LEVEL=WARNING \
  -- uvx awslabs.s3-mcp-server@latest

# Lambda — deploy and test functions
claude mcp add awslabs.lambda-mcp-server -s project \
  -e AWS_PROFILE="${AWS_PROFILE:-default}" \
  -e AWS_REGION="${AWS_REGION:-us-east-1}" \
  -e FASTMCP_LOG_LEVEL=WARNING \
  -- uvx awslabs.lambda-mcp-server@latest

# Bedrock — test embeddings and LLM
claude mcp add awslabs.bedrock-mcp-server -s project \
  -e AWS_PROFILE="${AWS_PROFILE:-default}" \
  -e AWS_REGION="${AWS_REGION:-us-east-1}" \
  -e FASTMCP_LOG_LEVEL=WARNING \
  -- uvx awslabs.bedrock-mcp-server@latest

echo ""

# ----------------------------------------------------------
# 3. Verify setup
# ----------------------------------------------------------
echo "✅ Setup complete! Verify with:"
echo "   claude mcp list"
echo ""
echo "📁 Project files created:"
echo "   .mcp.json    — MCP server configs (version controlled)"
echo "   CLAUDE.md    — Project instructions for Claude Code"
echo "   .env.example — Environment variable template"
echo ""
echo "🚀 Next steps:"
echo "   1. cp .env.example .env"
echo "   2. Edit .env with your AWS settings"
echo "   3. pip install -r requirements.txt"
echo "   4. Open Claude Code: claude"
echo "   5. Start building: 'Build the ingest Lambda handler'"
echo ""
