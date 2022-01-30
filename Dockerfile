FROM python:3.8-slim
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update
RUN apt-get -y install gcc
RUN pip install --trusted-host pypi.python.org -r requirements.txt
COPY . /app
COPY data/ratings.csv ./data/ratings.csv
COPY data/movies.csv ./data/movies.csv
COPY model/model.pickle ./model/model.pickle
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]