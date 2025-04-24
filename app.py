import os
import getpass
from dotenv import load_dotenv

from langsmith import utils
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

is_tracing_enabled = utils.tracing_is_enabled()

if is_tracing_enabled:
    # init the research tool
    search = TavilySearchResults(max_results=2)

    # add it to the list of tools
    tools = [search]

    # memory
    memory = MemorySaver()
    config = { "configurable": {"thread_id": "chat1"} }

    # model
    model = ChatGroq(model_name="Deepseek-R1-Distill-Qwen-32b", groq_api_key=groq_api_key)
    
    # bind with the research tool
    # model_with_tools = model.bind_tools(tools)

    # create the agent which will under the hood bind_tools to the model
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # streaming
    for step in agent_executor.stream({"messages": [HumanMessage(content="Hi i'm manuela")]}, config=config):
        print(step)
        print("----------")

    print("******************")
    for step in agent_executor.stream({"messages": [HumanMessage(content="what's my name")]}, config=config):
        print(step)
        print("----------")