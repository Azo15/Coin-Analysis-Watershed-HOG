FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir opencv-python-headless fastapi uvicorn
COPY para_analizi.py demo.py demo.html 1.jpg 2.jpg 3.jpg ./
RUN python para_analizi.py && mkdir samples && mv 1.jpg 2.jpg 3.jpg guncel_sonuc_*.jpg samples/
ENV PORT=8000
CMD ["sh", "-c", "exec uvicorn demo:app --host 0.0.0.0 --port \"$PORT\""]
