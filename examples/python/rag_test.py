import os
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""

embedding = OpenAIEmbeddings(base_url=os.getenv("OPENAI_EMBEDDINGS_API_BASE"),
                              api_key=os.getenv("OPENAI_EMBEDDINGS_API_KEY"), 
                              model=os.getenv("OPENAI_EMBEDDINGS_API_MODEL"))

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_text(text)

client_settings = Settings(
    #chroma_api_impl="rest",
    chroma_server_host="localhost",
    #chroma_server_ssl_verify=False,
    chroma_server_http_port=8000
)

vector_db = Chroma.from_texts(texts, embedding, client_settings=client_settings, collection_name="rag_test_db")

llm = ChatOpenAI(model=os.getenv("OPENAI_API_MODEL"))

chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    verbose=True, 
    retriever=vector_db.as_retriever(),
)

question = """Who is Marie Curie?"""
answer = chain.invoke(question)

print("답변:", answer["result"], "\n")
