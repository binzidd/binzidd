#!/usr/bin/env bash
# deploy.sh – Build, push, and deploy the Month-End Assistant to ECS Fargate
#
# Usage:
#   ./infra/deploy.sh [env-file]
#
# Defaults to .env in the repo root if no argument is given.
#
# Prerequisites: aws-cli v2, docker, jq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${1:-${REPO_ROOT}/.env}"
STACK_NAME="month-end-assistant"
TEMPLATE_FILE="${SCRIPT_DIR}/cloudformation.yaml"
REGION="${AWS_REGION:-us-east-1}"

# ─── Colours ──────────────────────────────────────────────────────────────────
RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; RESET="\033[0m"

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

# ─── 1. Prerequisites ─────────────────────────────────────────────────────────
info "Checking prerequisites..."

command -v aws    >/dev/null 2>&1 || error "aws-cli v2 not found. Install from https://aws.amazon.com/cli/"
command -v docker >/dev/null 2>&1 || error "docker not found."
command -v jq     >/dev/null 2>&1 || error "jq not found. Install with: brew install jq"

AWS_VERSION=$(aws --version 2>&1 | head -1)
info "AWS CLI: ${AWS_VERSION}"

# ─── 2. Load .env ─────────────────────────────────────────────────────────────
if [[ -f "${ENV_FILE}" ]]; then
    info "Loading environment from ${ENV_FILE}"
    set -o allexport
    # shellcheck disable=SC1090
    source <(grep -E '^[A-Z_]+=.*' "${ENV_FILE}" | sed 's/#.*//' | grep -v '^$')
    set +o allexport
else
    warn "No env file found at ${ENV_FILE} -- using environment variables only"
fi

REGION="${AWS_REGION:-${REGION}}"

# ─── 3. Validate AWS credentials ──────────────────────────────────────────────
info "Validating AWS credentials..."
CALLER_IDENTITY=$(aws sts get-caller-identity --region "${REGION}" 2>&1) || \
    error "AWS credentials invalid or not configured.\n${CALLER_IDENTITY}"

ACCOUNT_ID=$(echo "${CALLER_IDENTITY}" | jq -r '.Account')
info "AWS Account: ${ACCOUNT_ID} | Region: ${REGION}"
success "Credentials OK"

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ─── 4. Authenticate Docker to ECR ────────────────────────────────────────────
info "Authenticating Docker to ECR..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_BASE}"

# ─── Helper: ensure ECR repo exists ───────────────────────────────────────────
ensure_ecr_repo() {
    local repo_name="$1"
    if ! aws ecr describe-repositories \
            --repository-names "${repo_name}" \
            --region "${REGION}" >/dev/null 2>&1; then
        info "Creating ECR repository ${repo_name}..."
        aws ecr create-repository \
            --repository-name "${repo_name}" \
            --image-scanning-configuration scanOnPush=true \
            --region "${REGION}"
        success "Repository created: ${ECR_BASE}/${repo_name}"
    else
        success "Repository exists:  ${ECR_BASE}/${repo_name}"
    fi
}

# ─── 5. Backend: build + push ─────────────────────────────────────────────────
BACKEND_REPO="month-end-backend"
ensure_ecr_repo "${BACKEND_REPO}"

info "Building backend image (linux/arm64)..."
docker build \
    --platform linux/arm64 \
    -t "month-end-backend:latest" \
    -f "${REPO_ROOT}/frontend/Dockerfile.backend" \
    "${REPO_ROOT}"
success "Backend image built"

IMAGE_TAG="$(date +%Y%m%d%H%M%S)"
BACKEND_URI="${ECR_BASE}/${BACKEND_REPO}"
BACKEND_FULL="${BACKEND_URI}:${IMAGE_TAG}"

docker tag "month-end-backend:latest" "${BACKEND_FULL}"
docker tag "month-end-backend:latest" "${BACKEND_URI}:latest"

info "Pushing backend image..."
docker push "${BACKEND_FULL}"
docker push "${BACKEND_URI}:latest"
success "Backend pushed: ${BACKEND_FULL}"

# ─── 6. Streamlit: build + push ───────────────────────────────────────────────
STREAMLIT_REPO="month-end-streamlit"
ensure_ecr_repo "${STREAMLIT_REPO}"

info "Building Streamlit image (linux/arm64)..."
docker build \
    --platform linux/arm64 \
    -t "month-end-streamlit:latest" \
    -f "${REPO_ROOT}/frontend/Dockerfile.streamlit" \
    "${REPO_ROOT}"
success "Streamlit image built"

STREAMLIT_URI="${ECR_BASE}/${STREAMLIT_REPO}"
STREAMLIT_FULL="${STREAMLIT_URI}:${IMAGE_TAG}"

docker tag "month-end-streamlit:latest" "${STREAMLIT_FULL}"
docker tag "month-end-streamlit:latest" "${STREAMLIT_URI}:latest"

info "Pushing Streamlit image..."
docker push "${STREAMLIT_FULL}"
docker push "${STREAMLIT_URI}:latest"
success "Streamlit pushed: ${STREAMLIT_FULL}"

# ─── 7. Auto-detect default VPC + 2 public subnets ───────────────────────────
info "Detecting default VPC and public subnets..."

VPC_ID="${VPC_ID:-}"
SUBNET_ID1="${SUBNET_ID1:-}"
SUBNET_ID2="${SUBNET_ID2:-}"

if [[ -z "${VPC_ID}" ]]; then
    VPC_ID=$(aws ec2 describe-vpcs \
        --filters "Name=is-default,Values=true" \
        --query "Vpcs[0].VpcId" \
        --output text \
        --region "${REGION}")
    [[ "${VPC_ID}" == "None" || -z "${VPC_ID}" ]] && \
        error "No default VPC found. Set VPC_ID in your env file."
    info "Default VPC: ${VPC_ID}"
fi

if [[ -z "${SUBNET_ID1}" || -z "${SUBNET_ID2}" ]]; then
    SUBNETS_JSON=$(aws ec2 describe-subnets \
        --filters \
            "Name=vpc-id,Values=${VPC_ID}" \
            "Name=map-public-ip-on-launch,Values=true" \
        --query "Subnets[*].{Id:SubnetId,Az:AvailabilityZone}" \
        --output json \
        --region "${REGION}")

    SUBNET_COUNT=$(echo "${SUBNETS_JSON}" | jq 'length')
    if [[ "${SUBNET_COUNT}" -lt 2 ]]; then
        warn "Fewer than 2 public subnets found; falling back to all subnets"
        SUBNETS_JSON=$(aws ec2 describe-subnets \
            --filters "Name=vpc-id,Values=${VPC_ID}" \
            --query "Subnets[*].{Id:SubnetId,Az:AvailabilityZone}" \
            --output json \
            --region "${REGION}")
        SUBNET_COUNT=$(echo "${SUBNETS_JSON}" | jq 'length')
        [[ "${SUBNET_COUNT}" -lt 2 ]] && \
            error "Need at least 2 subnets. Set SUBNET_ID1 and SUBNET_ID2 in your env file."
    fi

    SUBNET_ID1=$(echo "${SUBNETS_JSON}" | jq -r '.[0].Id')
    SUBNET_ID2=$(echo "${SUBNETS_JSON}" | jq -r '.[1].Id')
fi

info "Subnet 1: ${SUBNET_ID1}"
info "Subnet 2: ${SUBNET_ID2}"

# ─── 8. Deploy CloudFormation stack ──────────────────────────────────────────
info "Deploying CloudFormation stack '${STACK_NAME}'..."
info "Template: ${TEMPLATE_FILE}"

aws cloudformation deploy \
    --stack-name "${STACK_NAME}" \
    --template-file "${TEMPLATE_FILE}" \
    --region "${REGION}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        VpcId="${VPC_ID}" \
        SubnetId1="${SUBNET_ID1}" \
        SubnetId2="${SUBNET_ID2}" \
        BackendImage="${BACKEND_FULL}" \
        StreamlitImage="${STREAMLIT_FULL}" \
        AwsRegion="${REGION}" \
        BedrockModelId="${BEDROCK_MODEL_ID:-anthropic.claude-3-5-sonnet-20241022-v2:0}" \
        AgentCoreAgentId="${AGENTCORE_AGENT_ID:-}" \
        AgentCoreAgentAliasId="${AGENTCORE_AGENT_ALIAS_ID:-TSTALIASID}" \
        AgentCoreMemoryId="${AGENTCORE_MEMORY_ID:-}" \
        TeamsWebhookUrl="${TEAMS_WEBHOOK_URL:-}" \
        SlackBotToken="${SLACK_BOT_TOKEN:-}" \
        SlackApprovalChannel="${SLACK_APPROVAL_CHANNEL:-#month-end-approvals}" \
        HitlVarianceThresholdPct="${HITL_VARIANCE_THRESHOLD_PCT:-5}" \
        MaxResearchIterations="${MAX_RESEARCH_ITERATIONS:-3}" \
        SandboxEnabled="${SANDBOX_ENABLED:-true}" \
    --no-fail-on-empty-changeset

success "CloudFormation stack deployed"

# ─── 9. Print outputs ────────────────────────────────────────────────────────
info "Fetching stack outputs..."
OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs" \
    --output json)

APP_URL=$(echo "${OUTPUTS}"    | jq -r '.[] | select(.OutputKey=="AppUrl")        | .OutputValue')
BACKEND_URL=$(echo "${OUTPUTS}" | jq -r '.[] | select(.OutputKey=="BackendApiUrl") | .OutputValue')
HEALTH_URL=$(echo "${OUTPUTS}" | jq -r '.[] | select(.OutputKey=="HealthCheckUrl") | .OutputValue')

echo ""
echo -e "${GREEN}============================================================${RESET}"
echo -e "${GREEN}  Month-End Assistant deployed successfully!                ${RESET}"
echo -e "${GREEN}============================================================${RESET}"
echo ""
echo -e "  ${BLUE}Streamlit UI:${RESET}  ${APP_URL}"
echo -e "  ${BLUE}Backend API:${RESET}   ${BACKEND_URL}"
echo -e "  ${BLUE}Health:${RESET}        ${HEALTH_URL}"
echo ""
echo -e "${YELLOW}Note:${RESET} ECS tasks may take 1-2 minutes to become healthy."
echo -e "      Verify with:  curl ${HEALTH_URL}"
echo ""
