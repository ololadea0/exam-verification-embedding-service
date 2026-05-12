FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Railway provides PORT automatically
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]