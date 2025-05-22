import streamlit as st

# --- LangGraph Setup ---

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt
from state import State

load_dotenv()

async def human_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Call the interrupt function to pause the graph and handle user interaction.

    Once resumed, it will log the type of action which was returned from
    the interrupt function.
    """
    # Define the interrupt request
    action_request = ActionRequest(
        action="Decide Action",
        args={"args": "Choose what to do next"},
    )

    interrupt_config = HumanInterruptConfig(
        allow_ignore=True,  # Allow the user to `ignore` the interrupt.
        allow_respond=True,  # Allow the user to `respond` to the interrupt.
        allow_edit=True,  # Allow the user to `edit` the interrupt's args.
        allow_accept=True,  # Allow the user to `accept` the interrupt's args.
    )

    # The description will be rendered as markdown in the UI, so you may use markdown syntax.
    description = (
        "# Review Search Results\n"
        + "Please review the search results and decide how you would like to proceed.\n\n"
        + "- Accept: Use these search results as is\n"
        + "- Edit: Modify the search query to get different results\n" 
        + "- Ignore: Skip using these search results\n"
        + "- Respond: Provide feedback on what kind of results you're looking for"
    )

    request = HumanInterrupt(
        action_request=action_request, config=interrupt_config, description=description
    )

    human_response: HumanResponse = interrupt([request])[0]

    if human_response.get("type") == "response":
        message = f"User responded with: {human_response.get('args')}"
        return {"interrupt_response": message}
    elif human_response.get("type") == "accept":
        message = f"User accepted with: {human_response.get('args')}"
        return {"interrupt_response": message}
    elif human_response.get("type") == "edit":
        message = f"User edited with: {human_response.get('args')}"
        return {"interrupt_response": message}
    elif human_response.get("type") == "ignore":
        message = "User ignored interrupt."
        return {"interrupt_response": message}

    return {
        "messages": state["messages"][-1],
        "interrupt_response": "Unknown interrupt response type: " + str(human_response)
    }

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    else:
        return END


llm = init_chat_model("anthropic:claude-3-5-sonnet-20240620")

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=[tool])

# add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human_node", human_node)
graph_builder.add_node("tools", tool_node)

# add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    path_map={"tools": "tools", END: "human_node"},
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human_node", END)

graph = graph_builder.compile()
graph.name = "Agent Inbox Example"  


# --- Streamlit UI ---
st.set_page_config(page_title="LangGraph & Streamlit Chat", layout="centered", initial_sidebar_state="auto")

st.title("💬 Simple Chatbot powered by LangGraph & Streamlit")
st.markdown("""
This is a basic demonstration of a chatbot built using LangGraph for the backend logic 
and Streamlit for the user interface. 
Enter your message below and the LangGraph agent will respond.
""")

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm a simple LangGraph agent. Ask me something like 'hello', 'tell me a joke', or 'what is LangGraph?'"}
    ]

# Display existing chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat interface
if user_query := st.chat_input("Your message:", key="user_input_field"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare the initial state for the LangGraph application
    # The keys in this dictionary must match the keys defined in the GraphState TypedDict.
    initial_state_for_graph = {"messages": [user_query], "interrupt_response": ""} 

    # Invoke the LangGraph app with the initial state
    # The graph execution will update the state based on the defined nodes and edges.
    try:
        # `invoke` runs the graph from the entry point until it reaches END or a stopping condition.
        final_state = graph.invoke(initial_state_for_graph)
        assistant_response = final_state.get("agent_response", "Sorry, I encountered an issue and couldn't formulate a response.")
    except Exception as e:
        assistant_response = f"An error occurred while processing your request: {str(e)}"
        st.error(f"Error invoking LangGraph: {e}") # Show error in UI for better debugging

    # Add assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Optional: Add a footer or some information
st.sidebar.header("About")
st.sidebar.info(
    "This app demonstrates a basic integration of LangGraph for stateful agent logic "
    "and Streamlit for building the interactive web UI. "
    "The agent's responses are currently rule-based."
)
st.sidebar.markdown("---")
st.sidebar.markdown("To run this app:")
st.sidebar.code("pip install streamlit langgraph")
st.sidebar.code("streamlit run your_script_name.py")

