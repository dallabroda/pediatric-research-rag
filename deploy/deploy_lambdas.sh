#!/bin/bash
# Deploy Lambda functions for Pediatric Research RAG
#
# Prerequisites:
# - Run setup.sh first
# - Python 3.12 installed
# - pip installed
#
# Usage: ./deploy_lambdas.sh

set -e

# Load configuration
if [ ! -f deploy/config.env ]; then
    echo "Error: config.env not found. Run setup.sh first."
    exit 1
fi

source deploy/config.env

echo "======================================"
echo "Deploying Lambda Functions"
echo "======================================"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"
echo ""

# Create temp directory for packaging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Packaging Lambda functions..."

# Check if Docker layer exists (preferred for cross-platform builds)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$SCRIPT_DIR/layer.zip" ]; then
    echo "Using existing layer.zip from $SCRIPT_DIR/"
    cp "$SCRIPT_DIR/layer.zip" "$TEMP_DIR/layer.zip"
elif command -v docker &> /dev/null; then
    echo "Building Lambda layer with Docker (Linux binaries)..."
    "$PROJECT_ROOT/scripts/build_lambda_layer.sh"
    cp "$SCRIPT_DIR/layer.zip" "$TEMP_DIR/layer.zip"
else
    echo "WARNING: Docker not available. Building layer locally."
    echo "         This may cause issues if deploying from Windows/Mac."
    echo "         Install Docker and run scripts/build_lambda_layer.sh for Linux binaries."
    echo ""
    # Package dependencies locally (fallback)
    pip install -q -t "$TEMP_DIR/python" \
        boto3 \
        faiss-cpu \
        numpy \
        pypdf \
        pdfplumber
    cd "$TEMP_DIR"
    zip -q -r layer.zip python/
fi

# Copy Lambda code
echo "Copying Lambda code..."
cp -r lambdas "$TEMP_DIR/"
cp -r config "$TEMP_DIR/"

# Create Lambda layer with dependencies
echo "Publishing Lambda layer..."
cd "$TEMP_DIR"
LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name "pediatric-rag-deps" \
    --description "Dependencies for Pediatric Research RAG (Linux binaries)" \
    --compatible-runtimes python3.12 \
    --zip-file fileb://layer.zip \
    --region "$REGION" \
    --query 'LayerVersionArn' --output text)
echo "  Created layer: $LAYER_ARN"

# Function to create/update Lambda
deploy_lambda() {
    local FUNC_NAME=$1
    local HANDLER=$2
    local DESC=$3
    local TIMEOUT=${4:-60}
    local MEMORY=${5:-512}

    echo ""
    echo "Deploying $FUNC_NAME..."

    # Create zip
    cd "$TEMP_DIR"
    zip -q -r "${FUNC_NAME}.zip" lambdas/ config/

    # Check if function exists
    if aws lambda get-function --function-name "$FUNC_NAME" --region "$REGION" 2>/dev/null; then
        # Update existing function
        aws lambda update-function-code \
            --function-name "$FUNC_NAME" \
            --zip-file "fileb://${FUNC_NAME}.zip" \
            --region "$REGION" > /dev/null

        aws lambda update-function-configuration \
            --function-name "$FUNC_NAME" \
            --handler "$HANDLER" \
            --timeout "$TIMEOUT" \
            --memory-size "$MEMORY" \
            --layers "$LAYER_ARN" \
            --environment "Variables={S3_BUCKET=$BUCKET_NAME,EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0,LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0}" \
            --region "$REGION" > /dev/null

        echo "  Updated: $FUNC_NAME"
    else
        # Create new function
        aws lambda create-function \
            --function-name "$FUNC_NAME" \
            --runtime python3.12 \
            --role "$LAMBDA_ROLE_ARN" \
            --handler "$HANDLER" \
            --timeout "$TIMEOUT" \
            --memory-size "$MEMORY" \
            --zip-file "fileb://${FUNC_NAME}.zip" \
            --layers "$LAYER_ARN" \
            --environment "Variables={S3_BUCKET=$BUCKET_NAME,EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0,LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0}" \
            --region "$REGION" > /dev/null

        echo "  Created: $FUNC_NAME"
    fi

    # Return function ARN
    aws lambda get-function --function-name "$FUNC_NAME" --region "$REGION" \
        --query 'Configuration.FunctionArn' --output text
}

# Deploy functions
INGEST_ARN=$(deploy_lambda "pediatric-rag-ingest" "lambdas.ingest.handler.handler" "Document ingestion" 300 1024)
EMBED_ARN=$(deploy_lambda "pediatric-rag-embed" "lambdas.embed.handler.handler" "Embedding generation" 600 2048)
QUERY_ARN=$(deploy_lambda "pediatric-rag-query" "lambdas.query.handler.handler" "Query handling" 120 1024)
DOCS_ARN=$(deploy_lambda "pediatric-rag-documents" "lambdas.documents.handler.handler" "Document listing" 60 512)

# Configure S3 trigger for ingest function
echo ""
echo "Configuring S3 trigger..."

# Add permission for S3 to invoke Lambda
aws lambda add-permission \
    --function-name "pediatric-rag-ingest" \
    --statement-id "s3-trigger" \
    --action "lambda:InvokeFunction" \
    --principal "s3.amazonaws.com" \
    --source-arn "arn:aws:s3:::$BUCKET_NAME" \
    --source-account "$ACCOUNT_ID" \
    --region "$REGION" 2>/dev/null || true

# Configure S3 bucket notification
NOTIFICATION_CONFIG='{
    "LambdaFunctionConfigurations": [
        {
            "Id": "IngestTrigger",
            "LambdaFunctionArn": "'$INGEST_ARN'",
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {"Name": "prefix", "Value": "raw/"}
                    ]
                }
            }
        }
    ]
}'

aws s3api put-bucket-notification-configuration \
    --bucket "$BUCKET_NAME" \
    --notification-configuration "$NOTIFICATION_CONFIG"

echo "  Configured S3 trigger for raw/ prefix"

# Configure API Gateway integrations
echo ""
echo "Configuring API Gateway..."

# Helper function to setup API method
setup_api_method() {
    local RESOURCE_ID=$1
    local HTTP_METHOD=$2
    local LAMBDA_ARN=$3

    # Create method
    aws apigateway put-method \
        --rest-api-id "$API_ID" \
        --resource-id "$RESOURCE_ID" \
        --http-method "$HTTP_METHOD" \
        --authorization-type "NONE" \
        --region "$REGION" 2>/dev/null || true

    # Setup Lambda integration
    aws apigateway put-integration \
        --rest-api-id "$API_ID" \
        --resource-id "$RESOURCE_ID" \
        --http-method "$HTTP_METHOD" \
        --type "AWS_PROXY" \
        --integration-http-method "POST" \
        --uri "arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations" \
        --region "$REGION"

    # Add Lambda permission for API Gateway
    aws lambda add-permission \
        --function-name "${LAMBDA_ARN##*:}" \
        --statement-id "apigateway-${HTTP_METHOD}-${RESOURCE_ID}" \
        --action "lambda:InvokeFunction" \
        --principal "apigateway.amazonaws.com" \
        --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/${HTTP_METHOD}/*" \
        --region "$REGION" 2>/dev/null || true
}

# POST /query -> query Lambda
setup_api_method "$QUERY_RESOURCE_ID" "POST" "$QUERY_ARN"
echo "  Configured POST /query"

# GET /documents -> documents Lambda
setup_api_method "$DOCS_RESOURCE_ID" "GET" "$DOCS_ARN"
echo "  Configured GET /documents"

# GET /documents/{id} -> documents Lambda
setup_api_method "$DOCS_ID_RESOURCE_ID" "GET" "$DOCS_ARN"
echo "  Configured GET /documents/{id}"

# Deploy API
echo ""
echo "Deploying API..."

aws apigateway create-deployment \
    --rest-api-id "$API_ID" \
    --stage-name "prod" \
    --description "Production deployment" \
    --region "$REGION" > /dev/null

API_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod"

echo ""
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Lambda Functions:"
echo "  - pediatric-rag-ingest (S3 trigger)"
echo "  - pediatric-rag-embed"
echo "  - pediatric-rag-query"
echo "  - pediatric-rag-documents"
echo ""
echo "API Endpoints:"
echo "  POST $API_URL/query"
echo "  GET  $API_URL/documents"
echo "  GET  $API_URL/documents/{id}"
echo ""
echo "Test with:"
echo "  curl -X POST $API_URL/query \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"What is pediatric ALL?\"}'"
echo ""
