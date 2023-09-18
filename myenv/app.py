from flask import Flask
from config import Config

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/')
def hello_world():
    return 'Hello, World!'
    

@app.route('/health')
def health_check():
    return 'OK'

if __name__ == '__main__':
    app.run()
