from langchain_core.tools import tool
from models.huggingface import Huggingface
from models.gpt import Gpt
from .base import Base


class Generic(Base):
    def is_active(self):
        return True

    def shall_terminate(self):
        return True

    def get_llm(self):
        """Return Qwen2.5-72B-Instruct as the LLM for this agent."""
        return Gpt().gpt_4o_mini()

    def get_system_message(self) -> str:
        """Return the system prompt instructing the LLM to handle directory operations."""
        return f"""You are a general-purpose agent."""

    def get_functionality(self) -> str:
        """Return a short description used by the Supervisor for routing decisions."""
        return 'handles user input that is not yet answered and does not need a specialized agent.'

    def get_agent_name(self) -> str:
        """Return the LangGraph node name for this agent: 'directory_agent'."""
        return "generic_agent"

    def get_tool_name(self) -> str:
        """Return the LangGraph node name for this agent's ToolNode: 'directory_tool'."""
        return "generic_tool"

    def get_applicable_tools(self) -> list:
        """Return the list of tools available to this agent."""
        return []
