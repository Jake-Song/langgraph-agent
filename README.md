# LangGraph Chatbot with Tools

A simple chatbot application built with LangGraph, LangChain, and Streamlit. The chatbot can use tools like web search (via Tavily) to answer questions.

## Features

- Interactive chat interface using Streamlit
- LangGraph workflow for agent orchestration
- Web search capabilities using Tavily Search
- OpenAI GPT-4 integration

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Running the Application

To start the chatbot, run:

```bash
streamlit run app.py
```

This will launch the Streamlit application in your default web browser.

## How It Works

The application uses LangGraph to define a workflow that:

1. Takes user input
2. Processes it with an LLM (GPT-4)
3. Uses tools like web search when needed
4. Returns a response to the user

The workflow is defined in `graph-test.py` and the web interface is implemented in `app.py`.
