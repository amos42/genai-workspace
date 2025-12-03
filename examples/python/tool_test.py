#%%capture --no-stderr
#%pip install python-dotenv bs4 langchain langchain-community langchain-openai


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage

load_dotenv()

@tool
def operator_sharp(a: int, b: int) -> int:
    """calcurate a # b.

    Args:
        a: first int
        b: second int
    """
    #print(f"call by llm operator # ({a}, {b})")
    return a * b

llm = ChatOpenAI(openai_api_base=os.getenv("OPENAI_API_BASE"), model_name=os.getenv("OPENAI_API_MODEL"))
result = llm.invoke("2 # 3 값은 몇이야? 한글로 답해 줘")
print("[tool 없을 때]")
print(result.content)

print("============================")

tools = [operator_sharp]
agent = create_agent(llm, tools=tools)

result = agent.invoke({"messages" : ["2 # 3 값은 몇이야?", "한글로 답해 줘"]})
# print(result)

content = ""
for msg in result["messages"]:
    if type(msg) is AIMessage:
        if msg.content: content += msg.content + "\n"
print("[tool 있을 때]")
print(content)
