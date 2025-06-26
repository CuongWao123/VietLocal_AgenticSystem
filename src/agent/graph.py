"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations


from typing import Any, Dict, List, cast , Literal
from datetime import datetime, timezone
from langchain_core.messages import AIMessage, AnyMessage
import json
# from react_agent.tools import TOOLS

from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent.Configuration import Configuration
from agent.State import State  
from agent.Tools import TOOLS1, TOOLS2
llm = ChatOpenAI(model = "gpt-4o-mini" , temperature=0.7)

# ── 3. Nodes -----------------------------------------------------------------
def supervisor(state: State) -> Dict[str, str]:
    """Route to agent_1 / agent_2 or stop."""
    history = state.messages

    resp: AIMessage = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    """
                        You are a supervisor managing two agents:
                        - Use agent_1 for general knowledge
                        - Use agent_2 for strategy/analysis
                        - Only return __end__ if the user query is fully answered.
                        - Assign work to one agent at a time, do not call agents in parallel.
                        - Do not do any work yourself. 
                        Return only a valid **JSON** object in this format:
                        {\"next_agent\": \"agent_1\" | \"agent_2\" | \"__end__\"}

            """
                ),
            },
            *history,
        ],
        response_format={"type": "json_object"},
    )

    next_agent = json.loads(resp.content)["next_agent"]
    # print(next_agent)  # Use logging if needed
    return {"next_agent": next_agent}   # <─ key for conditional edges


def agent_1(state: State) -> Dict[str, Any]:
    """General-knowledge responder."""
    history = state.messages
    llm_with_tools = llm.bind_tools(TOOLS1)
    reply: AIMessage = llm_with_tools.invoke(
        [
            {"role": "system", "content": "You are Agent 1, specialised in general knowledge. If the user asks for information that requires searching the web, always use the search web tool. Do not answer from your own knowledge if a tool is available. Respond with a tool call when appropriate."},
            *history,
        ]
    )
    return {"messages": [reply]}


def agent_2(state: State) -> Dict[str, Any]:
    """Analysis / strategy responder."""
    history = state.messages
    llm_with_tools = llm.bind_tools(TOOLS2)
    reply: AIMessage = llm_with_tools.invoke(
        [
            {"role": "system", "content": "You are Agent 2, specialised in analysis and strategy. If the user asks for information that requires database queries, always use the query database tool. Do not answer from your own knowledge if a tool is available. Respond with a tool call when appropriate."},
            *history,
        ]
    )
    return {"messages": [reply]}


def route_agent1_tools(state: State) -> str:
    """Determine the next node based on the agent1's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("supervisor" or "tools1").
    """
    last_message = state.messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return "supervisor"  # No tool calls, return to supervisor
    return "tools1"


def route_agent2_tools(state: State) -> str:
    """Determine the next node based on the agent2's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("supervisor" or "tools2").
    """
    last_message = state.messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return "supervisor"  # No tool calls, return to supervisor
    return "tools2"
    

# ── 4. Build the graph -------------------------------------------------------
builder = StateGraph(State)

# Add nodes
builder.add_node("supervisor", supervisor )
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)


# add tool nodes
builder.add_node("tools1", ToolNode(TOOLS1))
builder.add_node("tools2", ToolNode(TOOLS2)) 


# Static edges

builder.set_entry_point("supervisor")

# Add conditional edges for agents to handle tool calls
builder.add_conditional_edges(
    "agent_1",
    route_agent1_tools,
    {
        "supervisor": "supervisor",
        "tools1": "tools1",
    }
)

builder.add_conditional_edges(
    "agent_2",
    route_agent2_tools,
    {
        "supervisor": "supervisor", 
        "tools2": "tools2",
    }
)

# Tool nodes return to supervisor after execution
builder.add_edge("tools1", "agent_1")
builder.add_edge("tools2", "agent_2")




def routing_condition(state: State) -> str:
    return state.next_agent if hasattr(state, "next_agent") else "__end__"
    

# Conditional edges from supervisor
builder.add_conditional_edges(
    "supervisor",
    routing_condition
    ,
    {
        "agent_1": "agent_1",
        "agent_2": "agent_2",
        "__end__": END,
    }
)

graph = builder.compile(name="Supervisor-with-two-agents-edges")


