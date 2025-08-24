FROM python:3.10-slim

#install dependencies
RUN apt-get update && apt-get install -y git

#install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

#copy model and code 
COPY . /app
WORKDIR /app

#exposing the api 
EXPOSE 8000

#run the api

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]