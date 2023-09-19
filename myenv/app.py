from flask import Flask
from dotenv import load_dotenv
from quizGenrator import quizGenerator

# .env 파일 로드
load_dotenv()
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    database_uri = os.environ.get('DATABASE_URI')
    print(database_uri)

    return 'Hello, World!'

@app.route("/quiz")
def quiz():
    result = quizGenerator("rest api 가 뭔가요?")
    return result
    

@app.route("/health")
def health_check():
    return 'OK'

if __name__ == '__main__':
    app.run()
