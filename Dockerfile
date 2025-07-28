FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Run the main script
CMD ["python", "main.py"]
