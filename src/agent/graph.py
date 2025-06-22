"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations


from typing import Any, Dict, List, cast , Literal
from datetime import datetime, timezone
from langchain_core.messages import AIMessage, AnyMessage
import json

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from agent.Configuration import Configuration
from agent.State import State  
from agent.Tools import TOOLS  # Import tools if needed  

llm = ChatOpenAI(model = "gpt-4o-mini" , temperature=0.7)

# ── 3. Nodes -----------------------------------------------------------------
def supervisor(state: State) -> Literal["agent_1", "agent_2", "__end__"]:
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
    print(next_agent)
    return  {"next_agent" : next_agent}   # <─ key for conditional edges


def agent_1(state: State) -> Dict[str, Any]:
    """General-knowledge responder."""
    history = state.messages
    reply: AIMessage = llm.invoke(
        [
            {"role": "system", "content": "You are Agent 1, specialised in general knowledge."},
            *history,
        ]
    )
    return {"messages": [reply] , 
            "next_agent": "supervisor"
            }        # merged into state via add_messages


def agent_2(state: State) -> Dict[str, Any]:
    """Analysis / strategy responder."""
    history = state.messages
    reply: AIMessage = llm.invoke(
        [
            {"role": "system", "content": "You are Agent 2, specialised in analysis and strategy."},
            *history,
        ]
    )
    return {"messages": [reply],
            "next_agent": "supervisor"
            }


# ── 4. Build the graph -------------------------------------------------------
builder = StateGraph(State)

# Add nodes
builder.add_node("supervisor", supervisor )
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)

# Static edges

builder.set_entry_point("supervisor")
builder.add_edge("agent_1", "supervisor")
builder.add_edge("agent_2", "supervisor")

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


