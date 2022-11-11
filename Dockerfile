FROM python:3.10
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m pip3 install pymongo
RUN pip3 install --upgrade pip
COPY . /app
CMD uvicorn app:app --port 8080