# Multi-stage Docker build for DevStudio MCP Server
FROM python:3.11-slim AS base

# Install system dependencies for media processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    g++ \
    pkg-config \
    libffi-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./
COPY requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY devstudio_mcp/ ./devstudio_mcp/
COPY pyproject.toml ./
COPY README.md ./

# Install the package in development mode
RUN pip install -e .

# Create directories with proper permissions
RUN mkdir -p ./recordings ./temp ./logs ./config && \
    chown -R appuser:appuser ./recordings ./temp ./logs ./config

# Copy configuration files
COPY .env.example ./.env

# Switch to non-root user
USER appuser

# Expose the port (although MCP typically uses stdio)
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import devstudio_mcp; print('healthy')" || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MCP_LOG_LEVEL=INFO

# Default command to run the MCP server
CMD ["python", "-m", "devstudio_mcp.server"]

# Development stage with additional tools
FROM base AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Override command for development
CMD ["python", "-m", "devstudio_mcp.server"]

# Production stage (optimized)
FROM base AS production

# Remove unnecessary packages for smaller image
USER root
RUN apt-get update && apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER appuser

# Production configuration
ENV MCP_LOG_LEVEL=WARNING
ENV PYTHONOPTIMIZE=1

CMD ["python", "-m", "devstudio_mcp.server"]