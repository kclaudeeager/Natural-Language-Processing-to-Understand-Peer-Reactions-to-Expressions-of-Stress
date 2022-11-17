FROM python:3.10
WORKDIR /app
RUN pip3 install --upgrade pip
COPY requirements.txt /app/dependencies.txt
RUN pip install -r dependencies.txt


COPY . /app
CMD uvicorn app:app --port 8080