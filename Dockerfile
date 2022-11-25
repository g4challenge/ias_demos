FROM python:3.9-slim


EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY streamlit_app /app/

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit_mnist.py", "--server.port=8501", "--server.address=0.0.0.0"]