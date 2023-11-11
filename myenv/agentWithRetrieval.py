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
from langchain.agents.agent_toolkits import create_retriever_tool





OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent


class AgentConversation:
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
        self.retriever = docsearch.as_retriever(search_kwargs={'k':1})
        self.tool = create_retriever_tool(
        self.retriever, 
        "search_about_company_internal_works_and_reulations",
        "Searches and returns documents regarding the company internal work, regulations including information about vacation policies."
        )

        self.tools = [self.tool]
        system_message = SystemMessage(
        content=(
            "You are a chat assistant based on company information. "
            "If the question is not related to company work or regulations, you will respond with, '저는 사내 정보와 관련된 질문 외에는 답변할 수 없습니다.'"
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
        )
        )

        self.agent_executor = create_conversational_retrieval_agent(ChatOpenAI(temperature = 0), self.tools, verbose=True, system_message=system_message)

        self.qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(search_kwargs={'k':1}),memory=memory, return_source_documents=True, condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
)   
       
    def chat(self, input):
        
        response = self.agent_executor({"input": input.get("question")})
        
        #response = self.qa(input)
        #documents = response["source_documents"]
                
        #if documents and documents[0].dict().get('metadata').get('filename') is not None:
        #    return response["answer"] + "\n"+documents[0].dict().get('metadata').get('filename')+"에 근거하여 답변하였습니다."
        return response["output"]



'''
test =  AgentConversation()
test.initialize_pinecone()
test.initialize_docsearch()
test.chat({"question":"안녕 내이름은 밥"})
test.chat({"question":"내이름이 뭐라고?"})
test.chat({"question":"휴가는 어떻게 쓰나요?"})
test.chat({"question":"어디에 신청하면 되나요?"})
test.chat({"question":"독도는 어디 땅인가요?"})
'''