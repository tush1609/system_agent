from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state schema passed between all nodes in the LangGraph workflow.

    Attributes:
        messages: Full conversation history across all agent hops.
        next: Next agent/node name set by the Supervisor after each decision.
        final_answer: Resolved answer once the Supervisor decides FINISH.
        steps: max iteration permitted
    """
    messages: Annotated[List[BaseMessage], "Conversation history"]
    next: str
    final_answer: str
    steps: int
    done: bool

