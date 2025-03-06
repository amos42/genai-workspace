#%%capture --no-stderr
#%pip install python-dotenv bs4 langchain langchain-community langchain-openai

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=os.getenv("OPENAI_API_MODEL"))

# %%
# llm 다이렉트로 호출
result = llm.invoke("2 # 3 값은 몇이야?")
print(result)

# %%
from langchain_core.tools import tool
from pprint import pprint

@tool
def operator_sharp(a: int, b: int) -> int:
    """calcurate a # b.

    Args:
        a: first int
        b: second int
    """
    #print(f"call by llm operator # ({a}, {b})")
    return a * b

# %%

result = operator_sharp({"a":10, "b":20})
print(result)

# %%

llm_with_tools = llm.bind_tools([operator_sharp])
result = llm_with_tools.invoke("2 # 3 값은 몇이야?")
print(result)
#pprint(result.additional_kwargs)

# %%
# chain으로 호출
from operator import attrgetter
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

chain = prompt | llm_with_tools | attrgetter("tool_calls") | operator_sharp.map()
result = chain.invoke("2 # 3하면 몇이야?")
print(result, end="\n")
