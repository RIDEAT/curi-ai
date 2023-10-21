import pinecone
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI




OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

class MemoryConversation:
    def __init__(self):
        self.pinecone_api_key = PINECONE_API_KEY
        self.pinecone_api_env = PINECONE_API_ENV
        self.index_name = "chaintest"
        self.llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    def initialize_pinecone(self):
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_api_env)

    def initialize_docsearch(self):
        docsearch = Pinecone.from_existing_index(index_name=self.index_name, embedding=embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,input_key="question", output_key="answer")
        self.qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(search_kwargs={'k':1}),memory=memory, return_source_documents=True, condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
)   
       
    def chat(self, input):
        
        response = self.qa(input)
        #documents = response["source_documents"]
                
        #if documents and documents[0].dict().get('metadata').get('filename') is not None:
        #    return response["answer"] + "\n"+documents[0].dict().get('metadata').get('filename')+"에 근거하여 답변하였습니다."
        return response["answer"]




test =  MemoryConversation()
test.initialize_pinecone()
test.initialize_docsearch()

print(test.chat({"question":"휴가 신청 어떻게 하나요?"}))
print(test.chat({"question":"가기 며칠 전에 하면 돼?"}))


