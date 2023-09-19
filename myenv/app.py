from flask import Flask, jsonify, request
from dotenv import load_dotenv
from chatbot import chat
from pymongo import MongoClient



load_dotenv()
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
   

    return 'Hello, World!'


    

@app.route("/health")
def health_check():
    return 'OK'

    
@app.route('/chat', methods=['POST'])
def chat_to_ai():
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()

        # JSON 데이터 처리ß
        text = data.get('text')
        question = data.get('question')
        
        response = chat(text, question)

        return jsonify({'message': response}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
