#!/bin/bash
# Build Lambda layer with Linux-compatible binaries using Docker
# This solves the Windows binary issue when deploying from Windows/Mac

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
LAYER_NAME="pediatric-rag-deps"

echo "=== Building Lambda Layer with Docker ==="

# Ensure deploy directory exists
mkdir -p "$DEPLOY_DIR"

# Build the Docker image
echo "Building Docker image for Lambda layer..."
docker build -f "$PROJECT_ROOT/Dockerfile.lambda" -t lambda-layer-builder "$PROJECT_ROOT"

# Run container to extract layer zip
echo "Extracting layer.zip..."
docker run --rm -v "$DEPLOY_DIR:/output" lambda-layer-builder

# Verify the zip was created
if [ ! -f "$DEPLOY_DIR/layer.zip" ]; then
    echo "ERROR: layer.zip was not created"
    exit 1
fi

echo "Layer zip created: $DEPLOY_DIR/layer.zip"
echo "Size: $(du -h "$DEPLOY_DIR/layer.zip" | cut -f1)"

# Deploy to AWS if requested
if [ "$1" = "--deploy" ]; then
    echo ""
    echo "=== Deploying Layer to AWS ==="

    # Publish new layer version
    LAYER_ARN=$(aws lambda publish-layer-version \
        --layer-name "$LAYER_NAME" \
        --description "FAISS, numpy, boto3, pypdf, pdfplumber for pediatric-research-rag" \
        --zip-file "fileb://$DEPLOY_DIR/layer.zip" \
        --compatible-runtimes python3.12 \
        --query 'LayerVersionArn' \
        --output text)

    echo "Published layer: $LAYER_ARN"

    # Update Lambda functions to use new layer
    echo "Updating Lambda functions..."

    for FUNC in pediatric-rag-ingest pediatric-rag-embed pediatric-rag-query pediatric-rag-documents; do
        echo "  Updating $FUNC..."
        aws lambda update-function-configuration \
            --function-name "$FUNC" \
            --layers "$LAYER_ARN" \
            --query 'FunctionArn' \
            --output text 2>/dev/null || echo "  (function may not exist yet)"
    done

    echo ""
    echo "=== Layer deployment complete ==="
    echo "Layer ARN: $LAYER_ARN"
else
    echo ""
    echo "To deploy to AWS, run:"
    echo "  $0 --deploy"
fi
