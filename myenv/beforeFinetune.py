from langchain.llms import OpenAI
from langchain.chains import OpenAIModerationChain, SequentialChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

import openai

template = """Question: {question}

Answer:"""

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant in company"},
    {"role": "user", "content": "휴가 신청에 대해서 알려줘"}
  ]
)
print(completion.choices[0].message["content"])
