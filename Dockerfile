FROM python:3.13-bookworm

WORKDIR /app

# Update system packages for security patches
RUN apt-get update && apt-get upgrade -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]