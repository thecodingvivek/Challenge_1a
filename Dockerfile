# Use slim Python image, compatible with AMD64 CPU architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables for clean logs, no bytecode files, and no pip cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory inside the container
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files and folders
COPY run_system.py ./
COPY src ./src
COPY models/production/ultra_accuracy_optimized_classifier.pkl ./models/production/ultra_accuracy_optimized_classifier.pkl

# Create folders for input/output/results/models (if they don't exist)
RUN mkdir -p /app/input /app/output /app/models /app/results

# Set up a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command to run the system in "process" mode
CMD ["python", "run_system.py", "--mode", "process"]
