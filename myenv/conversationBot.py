from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv


load_dotenv()

loader = TextLoader("myenv/restApi.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,input_key="question", output_key="answer")

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(search_kwargs={'k':1}),memory=memory, return_source_documents=True, condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
)

query = "rest api가 뭐야"
result = qa({"question": query})

print(result["answer"])
print(result["source_documents"])

