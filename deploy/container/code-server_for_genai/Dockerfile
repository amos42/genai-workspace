FROM codercom/code-server:latest

#COPY ./__custom__.crt /etc/ssl/certs
#ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/__custom__.crt
#ENV CURL_CA_BUNDLE=/etc/ssl/certs/__custom__.crt

#ENV SSL_NO_VERIFY=1
#ENV NODE_TLS_REJECT_UNAUTHORIZED=1

RUN sudo apt update && \
    sudo apt install -y --fix-missing ca-certificates dnsutils python3-full python3-pip

#ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/__custom__.crt

RUN /usr/lib/code-server/bin/code-server --install-extension ms-python.python && \
    /usr/lib/code-server/bin/code-server --install-extension ms-toolsai.jupyter

WORKDIR /home/coder/workspace

COPY ./code-server_pip.conf /home/coder/.config/pip/pip.conf

RUN python3 -m venv .venv
RUN . .venv/bin/activate && \
    pip install ipykernel python-dotenv networkx \
                langgraph langchain langchain-community langchain_experimental \
                langchain-openai langchain-neo4j chromadb \
                neo4j_graphrag

COPY ./code-server_config.yaml /home/coder/.config/code-server/config.yaml

EXPOSE 8080
