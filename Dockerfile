FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install FastAPI and Uvicorn for API execution
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Expose port for Streamlit
EXPOSE 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
