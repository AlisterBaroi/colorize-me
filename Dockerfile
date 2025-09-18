# # Use Python --version: 3.11.8
# FROM python:3.11.8-alpine

# # Set working directory
# WORKDIR /app

# # Copy all files
# COPY . .

# # Install dependencies
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# # Run application (Linux)
# CMD ["streamlit", "run", "Colorize_Me.py", "--host", "0.0.0.0", "--port", "8080"]


# Use Debian slim so we get manylinux wheels for numpy/opencv/torch
FROM python:3.11-slim

# Prevent .pyc and ensure stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps needed at runtime by OpenCV etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Upgrade pip first; then install Python deps
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Cloud Run listens on $PORT; default to 8080
ENV PORT=8080

# Streamlit entrypoint
# CMD ["streamlit", "run", "Colorize_Me.py", "--server.address=0.0.0.0", "--server.port=8080"]
CMD ["streamlit", "run", "Colorize_Me.py", "--host", "0.0.0.0", "--port", "8080"]
