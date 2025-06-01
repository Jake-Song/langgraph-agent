"""Module for defining the agent's workflow graph and human interaction nodes."""

from langgraph.graph import StateGraph
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from state import State
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

load_dotenv()

llm = init_chat_model("openai:gpt-4.1")

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=[tool])

# Define a new graph
workflow = StateGraph(State)

# Add the node to the graph. This node will interrupt when it is invoked.
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `human_node` so the first node will interrupt
workflow.add_edge("__start__", "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
)
workflow.add_edge("tools", "chatbot")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Simple Agent Tools"  
