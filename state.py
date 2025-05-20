"""State module for managing agent state."""

from __future__ import annotations
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
