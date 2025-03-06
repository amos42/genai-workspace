import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(model=os.getenv("OPENAI_API_MODEL"))

question = """Who is Marie Curie?"""
answer = llm.invoke(question)
print("답변:", answer.content)
