#!/bin/bash
# Setup AWS resources for Pediatric Research RAG
#
# Creates:
# - S3 bucket for data and index storage
# - IAM role for Lambda execution
# - Lambda functions (ingest, embed, query, documents)
# - API Gateway REST API
#
# Prerequisites:
# - AWS CLI configured with appropriate credentials
# - jq installed for JSON parsing
#
# Usage: ./setup.sh [BUCKET_NAME] [REGION]

set -e

# Configuration
BUCKET_NAME="${1:-pediatric-research-rag}"
REGION="${2:-us-east-1}"
STACK_NAME="pediatric-research-rag"
LAMBDA_ROLE_NAME="${STACK_NAME}-lambda-role"
API_NAME="${STACK_NAME}-api"

echo "======================================"
echo "Pediatric Research RAG - AWS Setup"
echo "======================================"
echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo ""

# Check for required tools
command -v aws >/dev/null 2>&1 || { echo "AWS CLI required but not installed. Aborting." >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq required but not installed. Aborting." >&2; exit 1; }

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"
echo ""

# 1. Create S3 Bucket
echo "Creating S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "  Bucket already exists: $BUCKET_NAME"
else
    if [ "$REGION" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    else
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
    echo "  Created bucket: $BUCKET_NAME"
fi

# Create folder structure
echo "Creating bucket folder structure..."
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/" --content-length 0
aws s3api put-object --bucket "$BUCKET_NAME" --key "processed/chunks/" --content-length 0
aws s3api put-object --bucket "$BUCKET_NAME" --key "processed/index/" --content-length 0
echo "  Created folders: raw/, processed/chunks/, processed/index/"

# 2. Create IAM Role for Lambda
echo ""
echo "Creating IAM role for Lambda..."

TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}'

if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" 2>/dev/null; then
    echo "  Role already exists: $LAMBDA_ROLE_NAME"
else
    aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "Lambda execution role for Pediatric Research RAG"
    echo "  Created role: $LAMBDA_ROLE_NAME"
fi

LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

# Attach policies
echo "Attaching policies..."

# Basic Lambda execution
aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" 2>/dev/null || true

# S3 access
S3_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::'$BUCKET_NAME'",
                "arn:aws:s3:::'$BUCKET_NAME'/*"
            ]
        }
    ]
}'

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "${STACK_NAME}-s3-policy" \
    --policy-document "$S3_POLICY"

# Bedrock access
BEDROCK_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:'$REGION'::foundation-model/amazon.titan-embed-text-v2:0",
                "arn:aws:bedrock:'$REGION'::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                "arn:aws:bedrock:'$REGION'::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
            ]
        }
    ]
}'

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "${STACK_NAME}-bedrock-policy" \
    --policy-document "$BEDROCK_POLICY"

echo "  Attached S3 and Bedrock policies"

# Wait for role to propagate
echo "Waiting for IAM role to propagate..."
sleep 10

# 3. Create API Gateway REST API
echo ""
echo "Creating API Gateway..."

API_ID=$(aws apigateway get-rest-apis --query "items[?name=='$API_NAME'].id" --output text)

if [ -z "$API_ID" ]; then
    API_ID=$(aws apigateway create-rest-api \
        --name "$API_NAME" \
        --description "Pediatric Research RAG API" \
        --endpoint-configuration types=REGIONAL \
        --query 'id' --output text)
    echo "  Created API: $API_NAME (ID: $API_ID)"
else
    echo "  API already exists: $API_NAME (ID: $API_ID)"
fi

# Get root resource ID
ROOT_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --query 'items[?path==`/`].id' --output text)

# Create /query resource
echo "Creating API resources..."

QUERY_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --query "items[?path=='/query'].id" --output text)

if [ -z "$QUERY_RESOURCE_ID" ]; then
    QUERY_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id "$API_ID" \
        --parent-id "$ROOT_RESOURCE_ID" \
        --path-part "query" \
        --query 'id' --output text)
    echo "  Created /query resource"
fi

# Create /documents resource
DOCS_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --query "items[?path=='/documents'].id" --output text)

if [ -z "$DOCS_RESOURCE_ID" ]; then
    DOCS_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id "$API_ID" \
        --parent-id "$ROOT_RESOURCE_ID" \
        --path-part "documents" \
        --query 'id' --output text)
    echo "  Created /documents resource"
fi

# Create /documents/{id} resource
DOCS_ID_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --query "items[?path=='/documents/{id}'].id" --output text)

if [ -z "$DOCS_ID_RESOURCE_ID" ]; then
    DOCS_ID_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id "$API_ID" \
        --parent-id "$DOCS_RESOURCE_ID" \
        --path-part "{id}" \
        --query 'id' --output text)
    echo "  Created /documents/{id} resource"
fi

# Create /lineage resource
LINEAGE_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --query "items[?path=='/lineage'].id" --output text)

if [ -z "$LINEAGE_RESOURCE_ID" ]; then
    LINEAGE_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id "$API_ID" \
        --parent-id "$ROOT_RESOURCE_ID" \
        --path-part "lineage" \
        --query 'id' --output text)
    echo "  Created /lineage resource"
fi

# Create /lineage/{id} resource
LINEAGE_ID_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --query "items[?path=='/lineage/{id}'].id" --output text)

if [ -z "$LINEAGE_ID_RESOURCE_ID" ]; then
    LINEAGE_ID_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id "$API_ID" \
        --parent-id "$LINEAGE_RESOURCE_ID" \
        --path-part "{id}" \
        --query 'id' --output text)
    echo "  Created /lineage/{id} resource"
fi

# Save configuration for deploy script
echo ""
echo "Saving configuration..."

cat > deploy/config.env << EOF
# Generated by setup.sh - $(date)
BUCKET_NAME=$BUCKET_NAME
REGION=$REGION
ACCOUNT_ID=$ACCOUNT_ID
LAMBDA_ROLE_ARN=$LAMBDA_ROLE_ARN
API_ID=$API_ID
ROOT_RESOURCE_ID=$ROOT_RESOURCE_ID
QUERY_RESOURCE_ID=$QUERY_RESOURCE_ID
DOCS_RESOURCE_ID=$DOCS_RESOURCE_ID
DOCS_ID_RESOURCE_ID=$DOCS_ID_RESOURCE_ID
LINEAGE_RESOURCE_ID=$LINEAGE_RESOURCE_ID
LINEAGE_ID_RESOURCE_ID=$LINEAGE_ID_RESOURCE_ID
EOF

echo "  Saved to deploy/config.env"

# 4. Create SQS Dead Letter Queue
echo ""
echo "Creating Dead Letter Queue..."

DLQ_URL=$(aws sqs get-queue-url --queue-name "${STACK_NAME}-dlq" --query 'QueueUrl' --output text 2>/dev/null)

if [ -z "$DLQ_URL" ] || [ "$DLQ_URL" = "None" ]; then
    DLQ_URL=$(aws sqs create-queue \
        --queue-name "${STACK_NAME}-dlq" \
        --attributes '{"MessageRetentionPeriod":"1209600"}' \
        --query 'QueueUrl' --output text)
    echo "  Created DLQ: ${STACK_NAME}-dlq"
else
    echo "  DLQ already exists: ${STACK_NAME}-dlq"
fi

# Get DLQ ARN
DLQ_ARN=$(aws sqs get-queue-attributes \
    --queue-url "$DLQ_URL" \
    --attribute-names QueueArn \
    --query 'Attributes.QueueArn' --output text)

echo "  DLQ ARN: $DLQ_ARN"

# Add SQS permissions to Lambda role
SQS_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sqs:SendMessage",
                "sqs:GetQueueAttributes"
            ],
            "Resource": "'$DLQ_ARN'"
        }
    ]
}'

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "${STACK_NAME}-sqs-policy" \
    --policy-document "$SQS_POLICY"

echo "  Attached SQS policy to Lambda role"

# 5. Create EventBridge rule for scheduled refresh
echo ""
echo "Creating EventBridge rule for weekly refresh..."

RULE_NAME="${STACK_NAME}-weekly-refresh"

# Check if rule exists
RULE_EXISTS=$(aws events describe-rule --name "$RULE_NAME" 2>/dev/null)

if [ -z "$RULE_EXISTS" ]; then
    aws events put-rule \
        --name "$RULE_NAME" \
        --schedule-expression "rate(7 days)" \
        --state ENABLED \
        --description "Weekly data refresh for Pediatric Research RAG"
    echo "  Created EventBridge rule: $RULE_NAME"
else
    echo "  EventBridge rule already exists: $RULE_NAME"
fi

# Add EventBridge permissions to Lambda role
EVENTS_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "events:PutRule",
                "events:PutTargets",
                "events:RemoveTargets",
                "events:DeleteRule"
            ],
            "Resource": "arn:aws:events:'$REGION':'$ACCOUNT_ID':rule/'$RULE_NAME'"
        }
    ]
}'

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "${STACK_NAME}-events-policy" \
    --policy-document "$EVENTS_POLICY"

echo "  Attached EventBridge policy to Lambda role"

# 6. Add CloudWatch permissions for metrics
echo ""
echo "Adding CloudWatch metrics permissions..."

CLOUDWATCH_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "cloudwatch:namespace": "PediatricRAG"
                }
            }
        }
    ]
}'

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "${STACK_NAME}-cloudwatch-policy" \
    --policy-document "$CLOUDWATCH_POLICY"

echo "  Attached CloudWatch policy to Lambda role"

# Update config.env with new resources
cat >> deploy/config.env << EOF
DLQ_URL=$DLQ_URL
DLQ_ARN=$DLQ_ARN
REFRESH_RULE_NAME=$RULE_NAME
EOF

echo "  Updated deploy/config.env with DLQ and EventBridge config"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Run ./deploy_lambdas.sh to deploy Lambda functions"
echo "2. Run download scripts to populate data"
echo "3. Run seed_index.py to build the FAISS index"
echo ""
echo "Resources created:"
echo "  - S3 Bucket: $BUCKET_NAME"
echo "  - IAM Role: $LAMBDA_ROLE_NAME"
echo "  - API Gateway: $API_NAME"
echo ""
