from flask import Flask, jsonify, request
from dotenv import load_dotenv

from chatbotManager import ChatbotManager

from flask_cors import CORS


load_dotenv()
import os

app = Flask(__name__)
CORS(app,origins=["https://view.dev.workplug.team","https://view.workplug.team"])

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route("/health")
def health_check():
    return 'OK'

chatbotManager = ChatbotManager()

@app.route('/chatbot/<string:memberId>', methods=['POST'])
def chatbot_ai(memberId):
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()
        chatbot = chatbotManager.load_chatbot(user_id=memberId)
        response = chatbot.chat(data)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/chatbot/<string:memberId>', methods=['DELETE'])
def chatbot_reset(memberId):
    try:
        chatbotManager.delete_chatbot_session(user_id=memberId)
        return jsonify({'response': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    
    app.run(port=5050)
