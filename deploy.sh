#!/bin/bash

# DevStudio MCP Server Deployment Script
# Usage: ./deploy.sh [environment] [action]
# Environment: dev, staging, prod
# Action: build, start, stop, restart, logs

set -e

# Default values
ENVIRONMENT=${1:-dev}
ACTION=${2:-start}
PROJECT_NAME="devstudio-mcp"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker and docker-compose are installed
check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log_info "Dependencies check passed."
}

# Load environment variables
load_env() {
    local env_file=".env"

    if [ "$ENVIRONMENT" = "prod" ]; then
        env_file=".env.production"
    elif [ "$ENVIRONMENT" = "staging" ]; then
        env_file=".env.staging"
    fi

    if [ -f "$env_file" ]; then
        log_info "Loading environment from $env_file"
        export $(cat $env_file | xargs)
    else
        log_warn "Environment file $env_file not found. Using default values."
    fi
}

# Build images
build_images() {
    log_info "Building Docker images for $ENVIRONMENT environment..."

    if [ "$ENVIRONMENT" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml build --no-cache
    else
        docker-compose -f docker-compose.yml build --no-cache
    fi

    log_info "Build completed successfully."
}

# Start services
start_services() {
    log_info "Starting DevStudio MCP services for $ENVIRONMENT environment..."

    if [ "$ENVIRONMENT" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml up -d
    else
        docker-compose -f docker-compose.yml up -d
    fi

    log_info "Services started. Waiting for health checks..."
    sleep 10

    # Check service health
    check_health
}

# Stop services
stop_services() {
    log_info "Stopping DevStudio MCP services..."

    if [ "$ENVIRONMENT" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml down
    else
        docker-compose -f docker-compose.yml down
    fi

    log_info "Services stopped."
}

# Restart services
restart_services() {
    log_info "Restarting DevStudio MCP services..."
    stop_services
    sleep 5
    start_services
}

# Show logs
show_logs() {
    log_info "Showing logs for DevStudio MCP services..."

    if [ "$ENVIRONMENT" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml logs -f
    else
        docker-compose -f docker-compose.yml logs -f
    fi
}

# Check service health
check_health() {
    log_info "Checking service health..."

    # Wait for services to be ready
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "Up (healthy)"; then
            log_info "Services are healthy and ready!"
            return 0
        fi

        log_info "Attempt $attempt/$max_attempts: Waiting for services to be ready..."
        sleep 5
        ((attempt++))
    done

    log_error "Services failed to become healthy within timeout."
    show_service_status
    return 1
}

# Show service status
show_service_status() {
    log_info "Current service status:"

    if [ "$ENVIRONMENT" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml ps
    else
        docker-compose -f docker-compose.yml ps
    fi
}

# Clean up old images and containers
cleanup() {
    log_info "Cleaning up old Docker images and containers..."

    # Remove stopped containers
    docker container prune -f

    # Remove unused images
    docker image prune -f

    # Remove unused volumes (be careful in production)
    if [ "$ENVIRONMENT" != "prod" ]; then
        docker volume prune -f
    fi

    log_info "Cleanup completed."
}

# Run tests in container
run_tests() {
    log_info "Running tests in container..."

    docker-compose -f docker-compose.yml run --rm devstudio-mcp python -m pytest tests/ -v

    if [ $? -eq 0 ]; then
        log_info "All tests passed!"
    else
        log_error "Some tests failed."
        exit 1
    fi
}

# Main deployment logic
main() {
    log_info "DevStudio MCP Deployment Script"
    log_info "Environment: $ENVIRONMENT"
    log_info "Action: $ACTION"

    check_dependencies
    load_env

    case $ACTION in
        build)
            build_images
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        status)
            show_service_status
            ;;
        cleanup)
            cleanup
            ;;
        test)
            run_tests
            ;;
        deploy)
            build_images
            stop_services
            start_services
            ;;
        *)
            log_error "Unknown action: $ACTION"
            echo "Usage: $0 [environment] [action]"
            echo "Environment: dev, staging, prod"
            echo "Action: build, start, stop, restart, logs, status, cleanup, test, deploy"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"