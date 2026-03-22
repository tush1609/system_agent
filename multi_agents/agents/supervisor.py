from .base import Base
from . import agents_list
from models.gpt import Gpt
from multi_agents.agent_state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END
from logger import logger


class Supervisors(Base):
    def __init__(self):
        self.available_agents = ["FINISH"] + [x.get_agent_name() for x in agents_list]
        super().__init__()

    def is_active(self):
        return True

    def shall_terminate(self):
        return False

    def get_llm(self):
        """Return Qwen2.5-72B-Instruct as the LLM for this agent."""
        return Gpt().gpt_4o_mini()

    def get_system_message(self) -> str:
        description = [f'- {self.get_agent_name()} -> {self.get_functionality()}']
        description += [f'- {x.get_agent_name()} -> {x.get_functionality()},\n' for x in agents_list]

        return f"""
                    You are a supervisor that selects exactly one next step from the following options:
                    Available agents: {self.available_agents}
                    {description}
                    Your job is to route the latest unresolved user request to the most appropriate agent.
                    ROUTING RULES:
                    1. If the request clearly belongs to one specialist domain, choose that specialist agent.
                    2. If the request is broad, conversational, unclear, or does not clearly belong to any specialist, choose generic_agent.
                    3. If the request could fit multiple agents, choose the agent that should go first or the one that coordinates the core task.
                    4. Do not explain your reasoning.
                    OUTPUT FORMAT:
                    - Respond with EXACTLY one word from: {self.available_agents}. No explanation.
                    - Do not add punctuation, explanations, or extra text.
                """

    def get_functionality(self) -> str:
        """Return a short description used by the Supervisor for routing decisions."""
        return 'use only when the user’s request has already been fully handled and no further response is needed from any agent'

    def get_agent_name(self) -> str:
        """Return the LangGraph node name for this agent: 'directory_agent'."""
        return "supervisor"

    def get_tool_name(self) -> str:
        """Return the LangGraph node name for this agent's ToolNode: 'directory_tool'."""
        return ""

    def get_applicable_tools(self) -> list:
        """Return the list of tools available to this agent."""
        return []

    async def agent_executor(self, state: AgentState) -> Command:
        """
        Core supervisor execution step — decide which agent handles next.

        Reads the shared AgentState, filters conversation history to only
        relevant HumanMessage and AIMessage entries (skipping ToolMessages
        for a cleaner LLM context), then invokes the routing LLM to pick
        the next agent.

        Routing Decision Logic:
            1. Filter state messages to HumanMessage + AIMessage with content.
            2. Append system prompt + routing question.
            3. Invoke LLM → get decision string.
            4. Validate decision against available_agents list.
            5. If valid option found → goto that agent node.
            6. If LLM returns unexpected text → fallback to FINISH.
        """

        logger.debug("SUPERVISOR", state=state["messages"])
        msg_for_tool = [SystemMessage(content=self.get_system_message())] + state["messages"]
        logger.debug("SUPERVISOR", input_msg=msg_for_tool)

        response = self.get_llm().model().invoke(msg_for_tool)
        logger.debug("SUPERVISOR", response=response)

        decision = response.content.strip()

        if state["done"]:
            decision = "FINISH"
        elif state["steps"] == 4:
            await self.get_llm().stream_message("Query overload !!!!, you have confused me. Split it up or change it")
            decision = "FINISH"
        else:
            # for/else: else only runs if loop ends without break (no match found).
            for option in self.available_agents:
                if option in decision:
                    decision = option
                    break
            else:
                decision = "FINISH"   # fallback if LLM returned unexpected text

        logger.info("SUPERVISOR", decision=decision)

        return Command(
            update={"next": decision},
            goto=END if decision == "FINISH" else decision
        )
