from agentWithRetrieval import AgentConversation  # Assuming 'MemoryChatBot' is the correct class name

class ChatbotManager:
    def __init__(self):
        self.chatbot_sessions = {}

    def create_chatbot_session(self, user_id):
        chatbot = AgentConversation()
        chatbot.initialize_pinecone()
        chatbot.initialize_docsearch()
        self.chatbot_sessions[user_id] = chatbot

    def delete_chatbot_session(self, user_id):
        if user_id in self.chatbot_sessions:
            del self.chatbot_sessions[user_id]

    def load_chatbot(self, user_id):
        if user_id not in self.chatbot_sessions:
            self.create_chatbot_session(user_id)
        return self.chatbot_sessions[user_id]
    
'''
user_id = 'test'    
chatbot_manager = ChatbotManager()
chatbot = chatbot_manager.load_chatbot(user_id)
print(chatbot.chat({"question":"휴가는 어떻게 쓰나요?"}))
print(chatbot.chat({"question":"쓰기 며칠전에 말해야 하나요?"}))
'''