#!/usr/bin/env python
# coding: utf-8

# In[26]:



# In[23]:


# pip install langchain --upgrade
# Version: 0.0.164

# !pip install pypdf

# In[24]:


# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# ### Load your data

# In[27]:

"""
loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")

## Other options for loaders 
# loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

# In[28]:


data = loader.load()


# In[29]:


# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[30].page_content)} characters in your document')

# ### Chunk your data up into smaller documents

# In[30]:


# Note: If you're using PyPDFLoader then we'll be splitting for the 2nd time.
# This is optional, test out on your own data.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# In[31]:


print (f'Now you have {len(texts)} documents') """

# ### Create embeddings of your documents to get ready for semantic search

# In[34]:

txt_file_path = "myenv/restApi.txt"

# 텍스트 파일을 읽어서 내용을 저장할 리스트 생성
texts = []

# 텍스트 파일 열기
with open(txt_file_path, "r", encoding="utf-8") as txt_file:
    # 파일 내용을 한 줄씩 읽어서 리스트에 추가
    for line in txt_file:
        texts.append(line)


# In[35]:


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
import os

# In[36]:


# Check to see if there is an environment variable with you API keys, if not, use what you put below
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV') # You may need to switch with your env

# In[37]:


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# In[45]:


# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "chaintest" # put in the name of your pinecone index here

# In[42]:



# In[46]:
docsearch = Pinecone.from_texts([t for t in texts], embeddings, index_name=index_name)




# In[47]:


query = "rest api 가 뭔가요?"
docs = docsearch.similarity_search(query)

# In[49]:


# Here's an example of the first document that was returned
print(docs[1].page_content[:450])

# ### Query those docs to get your answer back

# In[50]:


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# In[51]:


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

# In[52]:

query = "rest api 가 뭔가요?"
docs = docsearch.similarity_search(query)

# In[53]:
chain.run(input_documents=docs, question=query)


def quizGenerator(input_data):
    # 이곳에서 원하는 작업을 수행하고 결과를 반환합니다.
    docs = docsearch.similarity_search(input_data)
    return chain.run(input_documents=docs ,question="Create 3 OX questions whose answer is O or X. follow the format: Q: Is seoul the capital of Korea? A: O")
