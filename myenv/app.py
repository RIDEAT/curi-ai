from flask import Flask, jsonify, request
from dotenv import load_dotenv
from chatbot import chat, create_index, modify_index, statueful_chat, statueful_chat_with_vector_db
from pymongo import MongoClient
from chatbotManager import ChatbotManager
from chatbotTest import read_s3_text_file
from flask_cors import CORS
from memoryAgentChain import chat_with_agent_chain, delete_agent_chain


load_dotenv()
import os

app = Flask(__name__)
CORS(app)

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
        question = data.get('question')
        
        response = chat(question)
        return response, 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
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

    
@app.route('/stateful-chat/<string:memberId>', methods=['POST'])
def stateful(memberId):
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()
                
        statueful_chat(data, memberId)
        return 'OK', 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500    
    

@app.route('/stateful-chat-vector/<string:memberId>', methods=['POST'])
def statefulVector(memberId):
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()
                
        statueful_chat_with_vector_db(data.get('question'), memberId)
        return 'OK', 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500      


@app.route('/stateful-chat-agent/<string:memberId>', methods=['POST'])
def statefulAgent(memberId):
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()
            
        return chat_with_agent_chain(data.get('question'), memberId), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500     
    

@app.route('/stateful-chat-agent/<string:memberId>', methods=['DELETE'])
def deleteContext(memberId):
    try:
        delete_agent_chain(memberId)
        return 'OK', 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500   
    


@app.route('/index', methods=['POST'])
def handle_index():
    try:
        # POST 요청에서 JSON 데이터 가져오기
        data = request.get_json()
        index = data.get('workspaceId')
        
        response = create_index(index)
        return response, 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/index', methods=['PUT'])
def put_index_content():
    try:
        # PUT 요청에서 TEXT 데이터 가져오기
        data = request.get_data().decode('utf-8')
        index = "chaintest"
        
        modify_index(index, data)
        return "OK", 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsert', methods=['POST'])
def chattest():
    try:
        response = read_s3_text_file()
        print(response)
        return response, 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500      
        

# feedback 이 왔을 때 지우는 것도 있어야함. 

if __name__ == '__main__':
    app.run(port=5050)
