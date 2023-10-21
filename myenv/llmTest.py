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
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory


from dotenv import load_dotenv

def chat(input):
    
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    
    index_name = "chaintest"
   
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    prompt_template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question. If you don't know from the context, then Just say that you don't know, don't try to make up an answer.:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer In Korean:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["history","context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT, "verbose": True, "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question")}

    workflow_search = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", verbose=True,retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, return_source_documents= True)
    

    response = workflow_search({"query": input})
    documents = response["source_documents"]
    if(documents[0].dict().get('metadata').get('filename') != None):
        print (response["result"] + documents[0].dict().get('metadata').get('filename'))
    
    else:
        print(response["result"])
    
    return
  
'''  
def chat(input):
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    
    index_name = "chaintest"

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    docs = docsearch.similarity_search(input.get('question'))
    return chain.run(input_documents=docs ,question=input.get('question'))
   
'''


'''
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
    llm_or_chain_factory=chat,
    evaluation=eval_config,
    verbose=True,
    project_name="llmchain-test-61",
)
'''

chat("rest api는 무엇인가요?")
chat("그것은 어디에 쓰이나요?")

