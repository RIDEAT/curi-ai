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
        workflowId = data.get('workflowId')

  # 파일 경로 생성
        file_path = os.path.join('myenv', 'workflow', f'{workflowId}.txt')

        # 파일에 텍스트 쓰기
        with open(file_path, 'w') as file:
            file.write(text)


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
        workflowId = data.get('workflowId')
        
        response = chat(text, workflowId)

        return jsonify({'message': response}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
