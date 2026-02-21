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
# 1. Check prerequisites
# ----------------------------------------------------------
echo "📋 Checking prerequisites..."

# Check for uv/uvx (required for MCP servers)
if ! command -v uvx &> /dev/null; then
    echo "  ⚠️  uvx not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  ✅ uv installed. Restart your terminal, then re-run this script."
    exit 1
else
    echo "  ✅ uvx found"
fi

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo "  ⚠️  AWS CLI not found. Please install it:"
    echo "     Windows: winget install Amazon.AWSCLI"
    echo "     macOS:   brew install awscli"
    echo "     Linux:   curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip' && unzip awscliv2.zip && sudo ./aws/install"
    exit 1
else
    echo "  ✅ AWS CLI found"
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "  ⚠️  AWS credentials not configured. Run: aws configure"
    exit 1
else
    echo "  ✅ AWS credentials configured"
fi

echo ""

# ----------------------------------------------------------
# 2. Add MCP Servers to project (if not already present)
# ----------------------------------------------------------
echo "🔌 Configuring MCP servers..."

# Check if .mcp.json already exists
if [ -f ".mcp.json" ]; then
    echo "  ✅ .mcp.json already exists"
else
    # AWS Documentation — look up Bedrock, Lambda, S3 APIs
    claude mcp add aws-documentation -s project \
      -e FASTMCP_LOG_LEVEL=WARNING \
      -- uvx awslabs.aws-documentation-mcp-server@latest

    # S3 — manage buckets and objects
    claude mcp add s3 -s project \
      -e AWS_PROFILE="${AWS_PROFILE:-default}" \
      -e AWS_REGION="${AWS_REGION:-us-east-1}" \
      -e FASTMCP_LOG_LEVEL=WARNING \
      -- uvx awslabs.s3-mcp-server@latest

    # Lambda — deploy and test functions
    claude mcp add lambda -s project \
      -e AWS_PROFILE="${AWS_PROFILE:-default}" \
      -e AWS_REGION="${AWS_REGION:-us-east-1}" \
      -e FASTMCP_LOG_LEVEL=WARNING \
      -- uvx awslabs.lambda-mcp-server@latest

    # Bedrock — test embeddings and LLM
    claude mcp add bedrock -s project \
      -e AWS_PROFILE="${AWS_PROFILE:-default}" \
      -e AWS_REGION="${AWS_REGION:-us-east-1}" \
      -e FASTMCP_LOG_LEVEL=WARNING \
      -- uvx awslabs.bedrock-mcp-server@latest
fi

echo ""

# ----------------------------------------------------------
# 3. Verify setup
# ----------------------------------------------------------
echo "✅ Setup complete! Verify MCP servers with:"
echo "   claude mcp list"
echo ""
echo "📁 Project files:"
echo "   .mcp.json    — MCP server configs (version controlled)"
echo "   CLAUDE.md    — Project instructions for Claude Code"
echo "   .env.example — Environment variable template"
echo ""
echo "🚀 Next steps:"
echo "   1. cp .env.example .env"
echo "   2. pip install -r requirements.txt"
echo "   3. Enable Bedrock models in AWS Console:"
echo "      - amazon.titan-embed-text-v2"
echo "      - anthropic.claude-3-haiku"
echo "      - anthropic.claude-3-sonnet"
echo "   4. Open Claude Code: claude"
echo "   5. Start building!"
echo ""
