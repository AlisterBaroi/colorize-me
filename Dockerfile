# Use Python --version: 3.11.8
FROM python:3.11.8-alpine

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run application (Linux)
CMD ["streamlit", "run", "Colorize_Me.py", "--host", "0.0.0.0", "--port", "8080"]