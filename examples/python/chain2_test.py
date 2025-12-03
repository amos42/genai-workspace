import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage
from dotenv import load_dotenv
import glob

load_dotenv()

@tool
def tool_list_files(path: str, pattern: str, recursive: bool) -> list[str]:
    """특정 디렉토리 속의 파일의 목록을 얻는다.

    Args:
        path: 디렉토리
        pattern: 와일드카드(*, ?, **)를 포함한 파일 목록 패턴. (예제. 서브디렉토리 포함시, './**/*.txt')
        recursive: 자식 디렉토리 포함 여부
    """

    return glob.glob(os.path.join(path, pattern), recursive=recursive)


llm = ChatOpenAI(openai_api_base=os.getenv("OPENAI_API_BASE"), model_name=os.getenv("OPENAI_API_MODEL"))

tools = [tool_list_files]
agent = create_agent(llm, tools=tools)
# agent = llm_with_tools | tools

result = agent.invoke({"messages" : ["d:/test 루트 디렉토리 속에 있는 확장자 py인 파일들의 목록을 알려 줘.", "한글로 답해 줘"]})
# print(result)

content = ""
for msg in result["messages"]:
    if type(msg) is AIMessage:
        content += msg.content + "\n"
print(content)
