from flask import Flask, jsonify, request
from dotenv import load_dotenv
from chatbot import chat, update
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

@app.route('/text-to-ai', methods=['POST'])
def text_to_ai():
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()

        text = data.get('text')
        
        update(text)


        return jsonify({'message': 'Data received and stored successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/chat', methods=['POST'])
def chat_to_ai():
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()

        # JSON 데이터 처리ß
        text = data.get('text')
        
        response = chat(text)

        return jsonify({'message': response}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
