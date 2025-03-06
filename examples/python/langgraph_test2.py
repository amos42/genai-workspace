# %%
# api key를 포함한 환경변수 세팅
import os
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(base_url=os.getenv("OPENAI_EMBEDDINGS_API_BASE"),
                              api_key=os.getenv("OPENAI_EMBEDDINGS_API_KEY"), 
                              model=os.getenv("OPENAI_EMBEDDINGS_API_MODEL"))

# %%
# 답변에 필요한 정보를 모아서 벡터DB를 구성한다.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from chromadb.config import Settings

# LLM 모델 생성
model = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_API_MODEL"), streaming=True)

urls = [
    # "https://www.google.com/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

client_settings = Settings(
    #chroma_api_impl="rest",
    chroma_server_host="localhost",
    #chroma_server_ssl_verify=False,
    chroma_server_http_port=8000
)

# 벡터 데이터베이스에 문서 추가
vectorstore = Chroma.from_documents(
    embedding=embedding,
    client_settings=client_settings,
    documents=doc_splits,
    collection_name="langgrah-test-db",
)

# %%
# 툴 정의
from langgraph.prebuilt import ToolNode
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 벡터DB의 정보를 얻기 위한 리트리버 생성
retriever = vectorstore.as_retriever()

# 릴리안 웡의 블로그 게시물에 대한 정보를 검색하고 반환하는 도구를 생성
tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [tool]

# 도구들을 실행할 ToolExecutor 객체를 생성
tool_executor = ToolNode(tools)

# %%
# 그래프 요소들을 정의 (Edge, Node)
import json
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

# 그래프 상태
class AgentState(TypedDict):
    # 메시지 시퀀스
    messages: Annotated[Sequence[BaseMessage], operator.add]

from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.output_parsers.openai_tools import PydanticToolsParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

# Edges

def should_retrieve(state):
    """
    에이전트가 더 많은 정보를 검색해야 하는지 또는 프로세스를 종료해야 하는지 결정

    이 함수는 상태의 마지막 메시지에서 함수 호출을 확인 함수 호출이 있으면 정보 검색 프로세스를 계속 그렇지 않으면 프로세스를 종료

    Args:
        state (messages): 현재 상태

    Returns:
        str: 검색 프로세스를 "계속"하거나 "종료"하는 결정
    """

    print("---DECIDE TO RETRIEVE---")
    messages = state["messages"]
    last_message = messages[-1]

    # 함수 호출이 없으면 종료
    if "tool_calls" not in last_message.additional_kwargs:
        print("---DECISION: DO NOT RETRIEVE / DONE---")
        return "end"
    # 그렇지 않으면 함수 호출이 있으므로 계속
    else:
        print("---DECISION: RETRIEVE---")
        return "continue"

def grade_documents_x(state):
    return "yes"

def grade_documents(state):
    """
    검색된 문서가 질문과 관련이 있는지 여부를 결정

    Args:
        state (messages): 현재 상태

    Returns:
        str: 문서가 관련이 있는지 여부에 대한 결정
    """

    print("---CHECK RELEVANCE---")

    # 데이터 모델
    class grade(BaseModel):
        """관련성 검사를 위한 이진 점수."""

        binary_score: str = Field(description="'yes' 또는 'no'의 관련성 점수")

    # 도구
    grade_tool_oai = convert_to_openai_tool(grade)

    # 도구와 강제 호출을 사용한 LLM
    llm_with_tool = model.bind(
        tools=[convert_to_openai_tool(grade_tool_oai)],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # 파서
    parser_tool = PydanticToolsParser(tools=[grade])

    # 프롬프트
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # 체인
    chain = prompt | llm_with_tool | parser_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    score = chain.invoke({"question": question, "context": docs})

    grade = score[0].binary_score

    if grade == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "yes"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(grade)
        return "no"

# Nodes

def agent(state):
    """
    현재 상태를 기반으로 에이전트 모델을 호출하여 응답을 생성 질문에 따라 검색 도구를 사용하여 검색을 결정하거나 단순히 종료

    Args:
        state (messages): 현재 상태

    Returns:
        dict: 메시지에 에이전트 응답이 추가된 업데이트된 상태
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    # model = ChatOpenAI(temperature=0, streaming=True,
    #                    model="gpt-4o")
    # 현재 사용 가능한 tool들을 모두 openai용 함수 형식으로 바꾸어 첨부한다.
    functions = [convert_to_openai_tool(t) for t in tools]
    llm_with_func = model.bind_tools(functions)    
    response = llm_with_func.invoke(messages)
    # 이것은 기존 목록에 추가될 것이므로 리스트를 반환.
    return {"messages": [response]}

def retrieve(state):
    """
    도구를 사용하여 검색을 실행

    Args:
        state (messages): 현재 상태

    Returns:
        dict: 검색된 문서가 추가된 업데이트된 상태
    """
    print("---EXECUTE RETRIEVAL---")
    messages = state["messages"]
    # 계속 조건을 기반으로 마지막 메시지가 함수 호출을 포함하고 있음을 알 수 있습니다.
    last_message = messages[-1]

    tool_calls = last_message.additional_kwargs["tool_calls"]
    if len(tool_calls) > 0:
        tool_call = tool_calls[0]
        tool_name = tool_call["function"]["name"]
        tool_input = json.loads(tool_call["function"]["arguments"])

        tool = tool_executor.tools_by_name[tool_name]
        observation = tool.invoke(tool_input)
        function_message = ToolMessage(content=observation, tool_call_id=tool_call["id"])

        # 이것은 기존 목록에 추가될 것이므로 리스트를 반환
        return {"messages": [function_message]}

def rewrite(state):
    """
    질문을 변형하여 더 나은 질문을 생성

    Args:
        state (messages): 현재 상태

    Returns:
        dict: 재구성된 질문이 추가된 업데이트된 상태
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # 평가자
    # model = ChatOpenAI(
    #     temperature=0, model="gpt-4o", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    답변 생성

    Args:
        state (messages): 현재 상태

    Returns:
         dict: 재구성된 질문이 추가된 업데이트된 상태
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # 프롬프트
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # model = ChatOpenAI(model_name="gpt-4o",
    #                  temperature=0, streaming=True)

    # 후처리
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 체인
    rag_chain = prompt | model | StrOutputParser()

    # 실행
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# %%
# 그래프 생성
from langgraph.graph import END, StateGraph

# langgraph.graph에서 StateGraph와 END를 가져옵니다.
workflow = StateGraph(AgentState)

# 순환할 노드들을 정의
workflow.add_node("agent", agent)  # 에이전트 노드를 추가
workflow.add_node("retrieve", retrieve)  # 정보 검색 노드를 추가
workflow.add_node("rewrite", rewrite)  # 정보 재작성 노드를 추가
workflow.add_node("generate", generate)  # 정보 생성 노드를 추가

# 에이전트 노드 호출하여 검색 여부 결정
workflow.set_entry_point("agent")

# 검색 여부 결정
workflow.add_conditional_edges(
    "agent",
    # 에이전트 결정 평가
    should_retrieve,
    {
        # 도구 노드 호출
        "continue": "retrieve",
        "end": END,
    },
)

# `action` 노드 호출 후 진행될 경로
workflow.add_conditional_edges(
    "retrieve",
    # 에이전트 결정 평가
    grade_documents,
    {
        "yes": "generate",
        "no": "rewrite",
    },
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# 컴파일
app = workflow.compile()

# %%
# 그래프를 출력한다. (Jupyter Notebook에서만 유효)
from IPython.display import Image, display

try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass

# %%
# 그래프를 이용하여 질의
import pprint
from langchain_core.messages import HumanMessage

# HumanMessage 객체를 사용하여 질문 메시지를 정의
inputs = {
    "messages": [
        HumanMessage(
            content="What does Lilian Weng say about the types of agent memory?"
        )
    ]
}
# app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍
for output in app.stream(inputs):
    # 출력된 결과에서 키와 값을 순회
    for key, value in output.items():
        # 노드의 이름과 해당 노드에서 나온 출력을 출력
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        # 출력 값을 예쁘게 출력
        pprint.pprint(value, indent=2, width=80, depth=None)
    # 각 출력 사이에 구분선을 추가
    pprint.pprint("\n---\n")
