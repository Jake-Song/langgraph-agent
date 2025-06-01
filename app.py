"""Simple chatbot application using LangGraph and Streamlit."""

import streamlit as st
from langchain.schema import HumanMessage
from graph_test import graph


# Set up the Streamlit page
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("LangGraph Chatbot with Tools")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Get user input
user_input = st.chat_input("Ask me anything...")

# Process user input
if user_input:
    # Add user message to chat history
    user_message = HumanMessage(content=user_input)
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get response from the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Initialize the graph with the current messages
            config = {"configurable": {"thread_id": "1"}}
            state = {"messages": st.session_state.messages}
            
            # Run the graph
            result = graph.invoke(state, config=config)
            
            # Get the last message from the result
            ai_message = result["messages"][-1]
            st.session_state.messages.append(ai_message)
            
            # Display the AI response
            st.write(ai_message.content)

# Add a sidebar with information
with st.sidebar:
    st.subheader("About")
    st.write("""
    This is a simple chatbot built with:
    - LangGraph for the agent workflow
    - LangChain for LLM interactions
    - Streamlit for the web interface
    - Tavily Search for web search capabilities
    """)
    
    st.subheader("Instructions")
    st.write("""
    1. Type your question in the chat input
    2. The agent will respond and may use tools like web search when needed
    3. Continue the conversation naturally
    """)
