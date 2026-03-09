FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn pinecone_fastapi.main:app --host 127.0.0.1 --port 8000 & exec chainlit run chainlit_app.py --host 0.0.0.0 --port ${PORT:-8080}"]
