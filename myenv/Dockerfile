FROM python:3.9

COPY . /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
ENV FLASK_ENV=production


CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]