#!/bin/bash
# Teardown AWS resources for Pediatric Research RAG
#
# Deletes all AWS resources created by setup.sh and deploy_lambdas.sh
#
# Usage: ./teardown.sh [--force]
#
# Options:
#   --force   Skip confirmation prompt

set -e

FORCE=false
if [ "$1" = "--force" ]; then
    FORCE=true
fi

# Load configuration
if [ ! -f deploy/config.env ]; then
    echo "Error: config.env not found. Nothing to tear down."
    exit 1
fi

source deploy/config.env

echo "======================================"
echo "Pediatric Research RAG - Teardown"
echo "======================================"
echo ""
echo "This will delete the following resources:"
echo "  - Lambda functions: pediatric-rag-*"
echo "  - Lambda layer: pediatric-rag-deps"
echo "  - S3 bucket: $BUCKET_NAME (and all contents)"
echo "  - API Gateway: pediatric-research-rag-api"
echo "  - IAM role: pediatric-research-rag-lambda-role"
echo ""

if [ "$FORCE" = false ]; then
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "Deleting resources..."

# 1. Delete Lambda functions
echo "Deleting Lambda functions..."
for FUNC in pediatric-rag-ingest pediatric-rag-embed pediatric-rag-query pediatric-rag-documents; do
    if aws lambda get-function --function-name "$FUNC" --region "$REGION" 2>/dev/null; then
        aws lambda delete-function --function-name "$FUNC" --region "$REGION"
        echo "  Deleted: $FUNC"
    fi
done

# 2. Delete Lambda layer
echo "Deleting Lambda layer..."
LAYER_VERSIONS=$(aws lambda list-layer-versions \
    --layer-name "pediatric-rag-deps" \
    --region "$REGION" \
    --query 'LayerVersions[*].Version' --output text 2>/dev/null || echo "")

for VERSION in $LAYER_VERSIONS; do
    aws lambda delete-layer-version \
        --layer-name "pediatric-rag-deps" \
        --version-number "$VERSION" \
        --region "$REGION"
    echo "  Deleted layer version: $VERSION"
done

# 3. Delete API Gateway
echo "Deleting API Gateway..."
if [ -n "$API_ID" ]; then
    aws apigateway delete-rest-api --rest-api-id "$API_ID" --region "$REGION" 2>/dev/null || true
    echo "  Deleted API: $API_ID"
fi

# 4. Delete S3 bucket (must empty first)
echo "Deleting S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    # Empty the bucket
    aws s3 rm "s3://$BUCKET_NAME" --recursive 2>/dev/null || true
    # Delete the bucket
    aws s3api delete-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    echo "  Deleted bucket: $BUCKET_NAME"
fi

# 5. Delete IAM role
ROLE_NAME="pediatric-research-rag-lambda-role"
echo "Deleting IAM role..."
if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    # Detach managed policies
    for POLICY_ARN in $(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[*].PolicyArn' --output text); do
        aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
    done

    # Delete inline policies
    for POLICY_NAME in $(aws iam list-role-policies --role-name "$ROLE_NAME" --query 'PolicyNames' --output text); do
        aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "$POLICY_NAME"
    done

    # Delete role
    aws iam delete-role --role-name "$ROLE_NAME"
    echo "  Deleted role: $ROLE_NAME"
fi

# Remove config file
rm -f deploy/config.env
echo "  Removed config.env"

echo ""
echo "======================================"
echo "Teardown Complete!"
echo "======================================"
echo ""
echo "All AWS resources have been deleted."
echo ""
