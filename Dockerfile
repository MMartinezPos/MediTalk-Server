# Use Python 3.11.6 as the base image
FROM python:3.11.6-slim

# Set the working directory in the Docker image
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    portaudio19-dev \
    git

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy only the requirements file, to cache the pip install step separately
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Command to run on container start
CMD ["python", "app.py"]
