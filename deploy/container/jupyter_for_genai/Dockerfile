FROM quay.io/jupyter/scipy-notebook:2024-11-19
 
USER 0
RUN apt update && apt install -y ca-certificates dnsutils
USER 1000

COPY ./jupyter_pip.conf /home/jovyan/.config/pip/pip.conf

RUN pip install ipykernel python-dotenv networkx \
                langgraph langchain langchain-community langchain_experimental \
                langchain-openai langchain-neo4j chromadb \
                neo4j_graphrag

COPY ./jupyter_config.json /opt/conda/share/jupyter/lab/settings/overrides.json

EXPOSE 8888
