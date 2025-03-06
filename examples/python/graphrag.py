import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
#from langchain_google_vertexai import VertexAI 
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
#from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.graphs import NetworkxEntityGraph
from dotenv import load_dotenv

load_dotenv()


text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""

#llm = VertexAI(max_output_tokens=4000,model_name='text-bison-32k')
#llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(model=os.getenv("OPENAI_API_MODEL"))

documents = [Document(page_content=text)]
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
)
graph_documents = llm_transformer_filtered.convert_to_graph_documents(
    documents
)

graph = NetworkxEntityGraph()

# Add nodes to the graph
for node in graph_documents[0].nodes:
    print(f"* CREATE (:{node.type} {node.model_dump_json()})")
    graph.add_node(node.id)

# Add edges to the graph
for edge in graph_documents[0].relationships:
    print(f"* MERGE ({edge.source.id}) - [:{edge.type}] -> ({edge.target.id})")
    graph._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type,
        )

chain = GraphQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True
)

question = """Who is Marie Curie?"""
result = chain.invoke(question)

print("답변:", result["result"], "\n")
