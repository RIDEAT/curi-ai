import pinecone
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory




OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

class MemoryChatBot:
    def __init__(self):
        self.pinecone_api_key = PINECONE_API_KEY
        self.pinecone_api_env = PINECONE_API_ENV
        self.index_name = "chaintest"
        self.llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    def initialize_pinecone(self):
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_api_env)

    def initialize_docsearch(self):
        docsearch = Pinecone.from_existing_index(index_name=self.index_name, embedding=embeddings)
        prompt_template = """
        You are a company assistant chatbot that responds based on company's internal context (delimited by <ctx></ctx>).
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question (delimited by <qs></qs>).  
        Following example(delimted by <eg></eg>) contains 2 example questions (delimted by <eq></eq>) and answers (delimted by <ea></ea>).
        If context(delimited by <ctx></ctx>) is not related to the question(delimited by <qs></qs>), simply say "방금 말씀하신 것에 대해 워크플로우 내에 정보가 충분치 않습니다." 

        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        <eg>
        <eq>사내에서 이메일 보낼 때 조심해야 하는 것은 뭐야?</eq>
        <ea>사내 이메일 보낼 때 주의해야 할 사항은 기밀 정보 보호, 사회 공학 공격 방지, 정확한 수신자 명시, 업무용 이메일 계정 사용 등입니다. 보안 및 정확성을 유지하세요.</ea>
        
        <eq>Pull request를 날리고 며칠이 지나도록 허가가 안나면 어떡해?</eq>
        <ea>Pull request를 올린 후 3일간 허가를 기다리고, 그 기간 내에도 허가를 받지 못한 경우 자체적으로 커밋할 수 있습니다.</ea>
        </eg>
        ------
        <qs>
        {question}
        </qs>
        
        Answer In Korean: 
        """
        self.PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["history","context", "question"]
        )
        self.chain_type_kwargs = {"prompt": self.PROMPT, "verbose": True, "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question")}
        self.workflow_search = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",                                                
                                                           retriever=docsearch.as_retriever(
                                                        search_kwargs={'k':1}),
                                                           chain_type_kwargs=self.chain_type_kwargs,
                                                           return_source_documents=True)

    def chat(self, input):
        question = input.get('question')
        
        response = self.workflow_search({"query": question})
        documents = response["source_documents"]
        
        #document 가 도움이 된 경우에만 출처에 붙이자 . 
        
        if documents and documents[0].dict().get('metadata').get('filename') is not None:
            return response["result"] + "\n"+documents[0].dict().get('metadata').get('filename')+"에 근거하여 답변하였습니다."
        return response["result"]





