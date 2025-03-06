import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=os.getenv("OPENAI_API_MODEL"))

from langchain_core.tools import tool
from operator import attrgetter

@tool
def operator_xyz(a: int, b: int) -> int:
    """operate xzy a and b.

    Args:
        a: first int
        b: second int
    """
    print(f"call by llm operator xyz({a}, {b})")
    return a * b

llm_with_tools = llm.bind_tools([operator_xyz])
result = llm_with_tools.invoke("2과 3를 xyz 연산하면 몇이야?")
print(result)

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
prompt = PromptTemplate.from_template(template)

# from langchain.agents import create_openai_functions_agent
# #chain = prompt | llm_with_tools | attrgetter("tool_calls") | operator_xyz.map()
# chain = create_openai_functions_agent(llm, [llm_with_tools], prompt)
# #result = chain.invoke("2과 3를 xyz 연산하면 몇이야?")
# result = chain.invoke({"input": "2과 3를 xyz 연산하면 몇이야?"})
# #result = chain.invoke("안녕")
# print(result)

from langchain.agents import Tool, create_react_agent, AgentExecutor, AgentType

# 에이전트 초기화
agent = create_react_agent (
    llm,
    tools=[operator_xyz],  # 커스텀 도구를 리스트에 추가
    prompt=prompt
)

tools = [
    Tool(
        name="operator xyz",
        func=operator_xyz.run,
        description=operator_xyz.__doc__,
        verbose=True
    )
]

agent_executor = AgentExecutor(
    agent=agent,
    tools=[operator_xyz],
    verbose=True,
    return_intermediate_steps=True,
)

result = agent_executor.invoke({"input": "2과 3를 xyz 연산하면 몇이야?"})
print(result, end="\n")
