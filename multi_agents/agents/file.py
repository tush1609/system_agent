from langchain_core.tools import tool
from models.huggingface import Huggingface
from .base import Base
from logger import logger


class File(Base):
    """
    Specialist agent for file (filesystem) operations.

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
        """Return the system prompt instructing the LLM to handle file operations."""
        return f"""You are a specialized agent that can use tools for files operations to complete task.
                    If a required tool does not exist do not call any substitute tool, do not explain or ask for clarification. Simply stop and return an empty response."""

    def get_functionality(self) -> str:
        """Return a short description used by the Supervisor for routing decisions."""
        return 'handles files operations only.'

    def get_agent_name(self) -> str:
        """Return the LangGraph node name for this agent: 'file_agent'."""
        return "file_agent"

    def get_tool_name(self) -> str:
        """Return the LangGraph node name for this agent's ToolNode: 'file_tool'."""
        return "file_tool"

    def get_applicable_tools(self) -> list:
        """Return the list of tools available to this agent."""
        return [create_file, remove_file]


@tool
def create_file(file_path: str, file_name: str) -> str:
    """Tool to create a new file inside the given path and with given name."""
    print('creating file', file_path, file_name)
    logger.info('CREATE_FILE', file_path=file_path, file_name=file_name)

    return f"Created file at {file_path}/{file_name}"


@tool
def remove_file(file_path: str) -> str:
    """Remove the file at the specified path."""
    logger.info('REMOVE_FILE', file_path=file_path)

    return f"Removed file at {file_path}"
