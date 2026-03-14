from .base import Base
from . import agents_list
from models.gpt import Gpt
from multi_agents.agent_state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END


class Supervisors(Base):
    def __init__(self):
        self.available_agents = ["FINISH"] + [x.get_agent_name() for x in agents_list]
        super().__init__()

    def is_active(self):
        return True

    def get_llm(self):
        """Return Qwen2.5-72B-Instruct as the LLM for this agent."""
        return Gpt().gpt_4o_mini()

    def get_system_message(self) -> str:
        description = ['- FINISH -> if no other agent task, then it is finish']
        description += [f'- {x.get_agent_name()} -> {x.get_functionality()},\n' for x in agents_list]

        return f"""
                   You are a supervisor routing user queries to specialist agents.
                   Available agents: {self.available_agents}

                   {description}

                    IMPORTANT RULES:
                        - If the conversation history shows the task is ALREADY DONE, respond with FINISH
                        - If an AIMessage confirms task completion and no other task is left, respond with FINISH
                        - Only route to an agent if the task is NOT yet handled

                   Respond with EXACTLY one word from: {self.available_agents}
                   """

    def get_functionality(self) -> str:
        """Return a short description used by the Supervisor for routing decisions."""
        return 'supervision of node'

    def get_agent_name(self) -> str:
        """Return the LangGraph node name for this agent: 'directory_agent'."""
        return "supervisor"

    def get_tool_name(self) -> str:
        """Return the LangGraph node name for this agent's ToolNode: 'directory_tool'."""
        return ""

    def get_applicable_tools(self) -> list:
        """Return the list of tools available to this agent."""
        return []

    def agent_executor(self, state: AgentState) -> Command:
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
        print("\n[Supervisor] Deciding next agent...", state['messages'])

        # Filter to only human/AI messages with content — skip ToolMessages
        human_message_history = [
            HumanMessage(content=msg.content) if isinstance(msg, HumanMessage)
            else AIMessage(content=msg.content)
            for msg in state["messages"]
            if isinstance(msg, (HumanMessage, AIMessage)) and msg.content
        ]

        messages = human_message_history + [
            SystemMessage(content=self.get_system_message()),
            HumanMessage(content=f"\n\nBased on the interaction till now, Who should handle this?")
        ]

        print("Messages for supervisor ", messages)
        response = self.get_llm().model().invoke(messages)

        print('SUPERVISOR RESP ', response)
        decision = response.content.strip()
        print('SUPERVISOR decision ', decision)

        # for/else: else only runs if loop ends without break (no match found).
        for option in self.available_agents:
            if option in decision:
                decision = option
                break
        else:
            decision = "FINISH"   # fallback if LLM returned unexpected text

        print(f"[Supervisor] Routing to: {decision}")

        return Command(
            update={"next": decision},
            goto=END if decision == "FINISH" else decision
        )
