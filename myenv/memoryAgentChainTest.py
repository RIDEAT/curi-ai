import langsmith
from flask import Flask, jsonify, request

from langchain import chat_models, smith
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv


load_dotenv()

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

from langsmith import Client

client = Client()
dataset_name = "workspace-datasets"

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


datasets = client.list_datasets(dataset_name=dataset_name)
# In[35]:


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv


from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.utilities import SerpAPIWrapper

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

llm = OpenAI(temperature=0)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

index_name = "chaintest"

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Korean:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

workflow_search = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)



search = SerpAPIWrapper()

tools = [
     Tool(
        name="State of company workflow search",
        func=workflow_search.run,
        description="useful for when you need to answer questions about the company rules and knowledge about IT. Input should be a fully formed question.",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

# 딕셔너리를 사용하여 memberId에 해당하는 agent_chain을 저장
agent_chains = {}

def chat_with_agent_chain(query):
    try:
        # memberId에 해당하는 agent_chain을 가져옴
        memberId = "1"
        agent_chain = get_agent_chain(memberId)

        # agent_chain을 사용하여 대화
        return agent_chain.run(input=query.get('question'))

    except Exception as e:
        return str(e)

def get_agent_chain(memberId):
    if memberId not in agent_chains:
        # 새로운 agent_chain 생성
        agent_chain = create_agent_chain(memberId)
        agent_chains[memberId] = agent_chain
    return agent_chains[memberId]

def delete_agent_chain(memberId):
    agent_chains.pop(memberId)

# 새로운 agent_chain 생성 함수
def create_agent_chain(memberId):
    prefix = """Have a conversation with a human, answering the following questions as best you can. You should answer in Korean. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=OpenAI(temperature=0, max_tokens= 1000), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    
    

from langchain.smith import RunEvalConfig, run_on_dataset

eval_config = RunEvalConfig(
  evaluators=[
    # You can specify an evaluator by name/enum.
    # In this case, the default criterion is "helpfulness"
    "criteria",
    # Or you can configure the evaluator
    RunEvalConfig.Criteria("harmfulness"),
    RunEvalConfig.Criteria("conciseness"),

  ]
)
run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=chat_with_agent_chain,
    evaluation=eval_config,
    verbose=True,
    project_name="llmchain-test-13",
)

