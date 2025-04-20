FROM python:3.10-slim

WORKDIR /usr/src/app

# Copy requirements FIRST to leverage Docker cache
COPY api/requirements.txt ./api/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r api/requirements.txt

# explicitly copy everything else
COPY ./api ./api

# Explicitly set PYTHONPATH and working directory
ENV PYTHONPATH=/usr/src/app
WORKDIR /usr/src/app

# explicitly run command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

