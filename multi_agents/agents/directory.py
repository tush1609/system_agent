from langchain_core.tools import tool
from models.huggingface import Huggingface
from .base import Base
from logger import logger


class Directory(Base):
    """
    Specialist agent for directory (filesystem) operations.

    Inherits from Tool and self-registers with the Register singleton
    on instantiation, making it discoverable by the Supervisor.
    """

    def is_active(self):
        return True

    def shall_terminate(self):
        return False

    def get_llm(self):
        """Return Qwen2.5-72B-Instruct as the LLM for this agent."""
        return Huggingface().qwen_2_5_72b_instruct()

    def get_system_message(self) -> str:
        """Return the system prompt instructing the LLM to handle directory operations."""
        return f"""
            You are a specialized agent that can use tools for directory operations to complete task.
            If a required tool does not exist do not call any substitute tool, do not explain or ask for clarification. Simply stop and return an empty response."""

    def get_functionality(self) -> str:
        """Return a short description used by the Supervisor for routing decisions."""
        return 'handles directory operations only.'

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
    """Tool to create a new directory only inside the given path and with given name"""
    logger.info('CREATE_DIRECTORY', directory_path=directory_path, directory_name=directory_name)

    return f"Created directory at {directory_path}/{directory_name}"


@tool
def remove_directory(directory: str) -> str:
    """Remove the directory from the specified path"""
    logger.info('REMOVE_DIRECTORY', directory_path=directory)

    return f"Removed directory at {directory}"
