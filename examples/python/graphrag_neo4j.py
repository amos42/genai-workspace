# %%
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from neo4j_graphrag.retrievers import Text2CypherRetriever
from langchain_core.documents import Document
#from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# %%

#llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(model=os.getenv("OPENAI_API_MODEL"))

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""

documents = [Document(page_content=text)]
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    # allowed_nodes=["Person", "Country", "Organization"],
    # allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
)
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
    documents
)

# %%

#graph = Neo4jGraph(url="neo4j://neo4j.local:80", username="", password="")
graph = Neo4jGraph(url="neo4j://svc-neo4j:7687", username="", password="")

# Add nodes to the graph
graph.add_graph_documents(graph_documents_filtered)

retriever = Text2CypherRetriever(
    #driver=GraphDatabase.driver("neo4j://neo4j.local:80", auth=None),
    driver=GraphDatabase.driver("neo4j://svc-neo4j:7687", auth=None),
    llm=llm,  # type: ignore
    #neo4j_schema=neo4j_schema,
    #examples=examples,
)

# %%

chain = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True,
    allow_dangerous_requests=True,
    retriever=retriever,
)

question = """Who is Marie Curie?"""
result = chain.invoke(question)

print("답변:", result["result"], "\n")
