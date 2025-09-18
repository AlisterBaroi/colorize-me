# # # Use Python --version: 3.11.8
# # FROM python:3.11.8-alpine

# # # Set working directory
# # WORKDIR /app

# # # Copy all files
# # COPY . .

# # # Install dependencies
# # RUN pip install --no-cache-dir --upgrade -r requirements.txt

# # # Run application (Linux)
# # CMD ["streamlit", "run", "Colorize_Me.py", "--host", "0.0.0.0", "--port", "8080"]


# # Use Debian slim so we get manylinux wheels for numpy/opencv/torch
# FROM python:3.11-slim

# # Prevent .pyc and ensure stdout flushing
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# # System deps needed at runtime by OpenCV etc.
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     libgl1 \
# #     libglib2.0-0 \
# #     && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1 \
#     libglib2.0-0 \
#     curl \
#     ca-certificates \
#   && rm -rf /var/lib/apt/lists/*

# WORKDIR /app
# COPY . .

# # Fetch model files at build time (pinned URLs from OpenCV repos)
# # RUN mkdir -p models \
# #  && curl -L -o models/colorization_release_v2.caffemodel \
# #       # https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/colorization_release_v2.caffemodel \
# #       https://github.com/AlisterBaroi/colorize-me/raw/3c6a6de95ff58f755ec35364bd33a51cb748e822/models/colorization_release_v2.caffemodel \
# #  && ls -lh models


#  # Overwrite any LFS pointer with the real model binary (pin to a commit)
# # (Using raw.githubusercontent.com to avoid HTML redirects)
# # RUN curl -fsSL -o models/colorization_release_v2.caffemodel \
# #     https://raw.githubusercontent.com/AlisterBaroi/colorize-me/3c6a6de95ff58f755ec35364bd33a51cb748e822/models/colorization_release_v2.caffemodel \
# #  && python - <<'PY'
# # import os, sys
# # p = "models/colorization_release_v2.caffemodel"
# # s = os.path.getsize(p)
# # print(p, "size:", s, "bytes")
# # # Fail the build if it looks suspiciously small (e.g. an LFS pointer)
# # sys.exit(0 if s > 1_000_000 else 1)
# # PY


# # Ensure models dir, fetch the caffemodel, and verify it's not an LFS pointer
# RUN mkdir -p models \
#  && curl -fsSL -o models/colorization_release_v2.caffemodel \
#     https://raw.githubusercontent.com/AlisterBaroi/colorize-me/3c6a6de95ff58f755ec35364bd33a51cb748e822/models/colorization_release_v2.caffemodel \
#  && python -c "import os,sys; p='models/colorization_release_v2.caffemodel'; s=os.path.getsize(p); print(p,'size:',s,'bytes'); sys.exit(0 if s>1000000 else 1)"

# # Upgrade pip first; then install Python deps
# RUN pip install --upgrade pip \
#  && pip install --no-cache-dir -r requirements.txt

# # Cloud Run listens on $PORT; default to 8080
# ENV PORT=8080

# # Streamlit entrypoint
# CMD ["streamlit", "run", "Colorize_Me.py", "--server.address=0.0.0.0", "--server.port=8080"]
# # CMD ["streamlit", "run", "Colorize_Me.py", "--host", "0.0.0.0", "--port", "8080"]

# Use Debian slim so we get manylinux wheels for numpy/opencv/torch
FROM python:3.11-slim

# Prevent .pyc and ensure stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps needed at runtime by OpenCV etc. + curl to fetch model
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Ensure models dir, fetch the caffemodel, and verify it's not an LFS pointer
RUN mkdir -p models \
&& curl -fL --retry 5 --retry-connrefused --retry-delay 2 \
    -o models/colorization_release_v2.caffemodel \
    "https://github.com/AlisterBaroi/colorize-me/blob/3c6a6de95ff58f755ec35364bd33a51cb748e822/models/colorization_release_v2.caffemodel?raw=1" \
&& python -c "import os,sys; p='models/colorization_release_v2.caffemodel'; s=os.path.getsize(p); print(p,'size:',s,'bytes'); sys.exit(0 if s>1000000 else 1)"


# Cloud Run listens on $PORT; default to 8080
ENV PORT=8080

# Streamlit entrypoint
CMD ["streamlit", "run", "Colorize_Me.py", "--server.address=0.0.0.0", "--server.port=8080"]
