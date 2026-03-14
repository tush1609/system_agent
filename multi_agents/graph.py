from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from multi_agents.agents.supervisor import Supervisors
from multi_agents.agents import agents_list
from multi_agents.agent_state import AgentState

class Graph:
    """
    Assembles and compiles the full multi-agent LangGraph workflow.

    Reads all self-registered agents from the Register singleton and
    dynamically wires them into a StateGraph alongside the Supervisor.
    This means adding a new agent requires zero changes here — just
    instantiating the Tool subclass is enough.

    Graph Structure:
        START
          │
          ▼
       [supervisor]  ◄─────────────────────┐
          │                                │
          ▼                                │
       [agent_node]  ──► [tool_node] ──────┘
          │
       (FINISH)
          │
          ▼
         END
    """

    def __init__(self):
        """
        Build and compile the StateGraph by dynamically registering all
        agent and tool nodes from the Register singleton.

        For each registered agent, two nodes are added:
            - node[0] (agent_name) → agent_executor (LLM + routing logic)
            - node[1] (tool_name)  → tool_executor  (ToolNode execution)

        All routing between nodes is handled via Command.goto inside each
        node — no add_edge or add_conditional_edges needed beyond START.
        """
        workflow = StateGraph(AgentState)
        supervisor = Supervisors()

        # Add supervisor node
        workflow.add_node(supervisor.get_agent_name(), supervisor.agent_executor)

        # Dynamically add agent + tool nodes from Register
        for node in agents_list:
            if node.get_agent_name():
                workflow.add_node(node.get_agent_name(), node.agent_executor)  # agent_name → agent_executor
            if node.get_tool_name():
                workflow.add_node(node.get_tool_name(), node.tool_executor)   # tool_name  → tool_executor

        # Only one static edge needed — all other routing is via Command.goto
        workflow.add_edge(START, supervisor.get_agent_name())

        self.graph = workflow.compile()

    async def get_result(self, query: str) -> AgentState:
        """
        Invoke the compiled graph with a user query and return the final state.

        Wraps the query in a HumanMessage and initializes AgentState with
        empty defaults before passing it to the graph.

        """
        return await self.graph.ainvoke({
            "messages": [HumanMessage(content=query)],
            "next": "",
            "final_answer": ""
        })
