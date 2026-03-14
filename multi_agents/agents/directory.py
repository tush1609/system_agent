from pathlib import Path
from langchain_core.tools import tool
from models.huggingface import Huggingface
from .base import Base


class Directory(Base):
    """
    Specialist agent for directory (filesystem) operations.

    Inherits from Tool and self-registers with the Register singleton
    on instantiation, making it discoverable by the Supervisor.
    """

    def is_active(self):
        return True

    def get_llm(self):
        """Return Qwen2.5-72B-Instruct as the LLM for this agent."""
        return Huggingface().qwen_2_5_72b_instruct()

    def get_system_message(self) -> str:
        """Return the system prompt instructing the LLM to handle directory operations."""
        return "You are a linux expert. Use this agent to answer directory related operations. Return only the command whenever possible and keep your response short"

    def get_functionality(self) -> str:
        """Return a short description used by the Supervisor for routing decisions."""
        return 'create, update, delete directory'

    def get_agent_name(self) -> str:
        """Return the LangGraph node name for this agent: 'directory_agent'."""
        return "directory_agent"

    def get_tool_name(self) -> str:
        """Return the LangGraph node name for this agent's ToolNode: 'directory_tool'."""
        return "directory_tool"

    def get_applicable_tools(self) -> list:
        """Return the list of tools available to this agent."""
        return [create_directory, remove_directory]


@tool
def create_directory(directory_path: str, directory_name: str) -> str:
    """Create a new directory at the specified path(directory_path) having specified name(directory_name)."""
    print('creating directory', directory_path, directory_name)

    return f"Created directory at {directory_path}/{directory_name}"


@tool
def remove_directory(directory: str) -> str:
    """Remove the directory at the specified path."""
    print('removing directory', directory)

    return f"Removed directory at {directory}"
