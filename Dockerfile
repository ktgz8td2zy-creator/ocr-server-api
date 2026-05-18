FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY run.sh .
RUN chmod +x run.sh

ENV OCR_ENGINE=rapidocr

CMD ["./run.sh"]
