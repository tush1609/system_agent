from abc import ABC, abstractmethod
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from multi_agents.agent_state import AgentState


class Base(ABC):
    """
    Abstract base class for all agents in the multi-agent hierarchy.

    Each agent in the system inherits from this class and implements
    the abstract methods to define its identity, LLM, system prompt,
    and tools. The base class handles all boilerplate for tool binding,
    tool execution, and LangGraph Command routing automatically.

    On instantiation, every agent self-registers with the global Register
    singleton, making it discoverable by the Supervisor for routing.

    Design Patterns:
        - Template Method Pattern: agent_executor() and tool_executor()
          provide fixed execution skeletons; subclasses only define what
          varies (LLM, tools, system message, names).
        - Self-Registration Pattern: __init__ calls Register.get_instance()
          so agents are auto-discovered without manual wiring.

    """

    def __init__(self):
        """
        Initialize the agent, resolve all abstract properties, and
        self-register with the global Register singleton.

        Note:
            Abstract methods must be implemented before __init__ is called.
            Python will raise TypeError if any abstract method is missing.
        """
        self.__tool_name = self.get_tool_name()
        self.__agent_name = self.get_agent_name()
        self.__system_message = self.get_system_message()
        self.__applicable_tools = self.get_applicable_tools()

    @abstractmethod
    def is_active(self):
        """
        Determine whether this tool is active.
        """
        pass

    @abstractmethod
    def get_llm(self):
        """
        Return the configured LLM instance for this agent.

        Use any Models subclass (Gpt, Huggingface) with a selected model.
        Called internally by get_bind_tools() to bind tools to the LLM.
        """
        pass

    @abstractmethod
    def get_system_message(self) -> str:
        """
        Return the system prompt for this agent.

        This message is prepended to every invocation to set the agent's
        persona and instruct it on how to behave and use its tools.
        """
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the unique node name for this agent in the LangGraph graph.

        This name is used as the LangGraph node key and as the goto target
        when tool_executor routes back to this agent after tool execution.
        """
        pass

    @abstractmethod
    def get_tool_name(self) -> str:
        """
        Return the unique node name for this agent's tool executor node
        in the LangGraph graph.

        This name is used as the LangGraph node key for the ToolNode and
        as the goto target when agent_executor detects tool calls.
        """
        pass

    @abstractmethod
    def get_functionality(self) -> str:
        """
        Return a plain-English description of what this agent handles.

        This description is passed to the Supervisor's system prompt so
        the LLM knows when to route queries to this agent.
        """
        pass

    @abstractmethod
    def get_applicable_tools(self) -> list[BaseTool]:
        """
        Return the list of @tool-decorated functions this agent can use.

        These tools are bound to the LLM via get_bind_tools() and passed
        to ToolNode via get_tool_node() for execution.
        """
        pass

    async def agent_executor(self, state: AgentState) -> Command:
        """
        Core agent execution step — invoke the LLM and decide next action.

        Prepends the system message to the conversation history, invokes
        the tool-bound LLM, and routes via Command.goto:
          - If the LLM emits tool_calls → goto tool_name (run the tools)
          - If no tool_calls → goto 'supervisor' (task complete for now)

        Args:
            state (AgentState): Current shared graph state containing
                                the full message history.

        Returns:
            Command: A LangGraph Command containing the updated messages
                     and the next node to jump to.
        """

        msg_for_tool = state["messages"]

        if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
            msg_for_tool = [SystemMessage(content=self.get_system_message())] + state["messages"]

        content, tools = await self.get_llm().tool_response(msg_for_tool, self.get_applicable_tools())
        msg_for_tool += [AIMessage(content=content, tool_calls=tools)]

        if tools and len(tools) > 0:
            return Command(
                update={"messages": msg_for_tool},
                goto=self.get_tool_name()      # jump to tool executor node
            )

        return Command(
            update={"messages": msg_for_tool},
            goto="supervisor"            # return to supervisor for next routing decision
        )

    async def tool_executor(self, state: AgentState) -> Command:
        """
        Tool execution step — run all pending tool calls and loop back.

        Invokes the ToolNode with the current message history (which
        contains the LLM's tool_calls), collects ToolMessages from each
        tool, appends them to the state, and routes back to this agent
        so it can process the tool results and generate a final response.

        Args:
            state (AgentState): Current shared graph state containing
                                the LLM response with tool_calls.

        Returns:
            Command: A LangGraph Command containing the updated messages
                     (with tool results appended) and goto agent_name to
                     loop back to this agent.
        """

        result = await self.get_llm().tool_execution_response(ToolNode(self.get_applicable_tools()), {"messages": state["messages"]})

        return Command(
            update={"messages": state["messages"] + result},
            goto=self.get_agent_name()         # loop back to agent to process tool results
        )
