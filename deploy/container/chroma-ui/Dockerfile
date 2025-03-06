FROM node:22

RUN git config --global http.sslVerify false

WORKDIR /app

RUN git clone https://github.com/thakkaryash94/chroma-ui

WORKDIR /app/chroma-ui

RUN sed -i 's|http://127.0.0.1:8000|http://chromadb.local:80|g' app/page.tsx

RUN npm --global config set strict-ssl false
RUN npm install

EXPOSE 3000

CMD [ "npm", "run", "dev" ]
