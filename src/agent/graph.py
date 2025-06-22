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

from agent.Configuration import Configuration
from agent.State import State  
from agent.Tools import TOOLS  # Import tools if needed  

llm = ChatOpenAI(model = "gpt-4o-mini" , temperature=0.7)


def supervisor(state: State) -> Command[Literal["agent_1", "agent_2", END]]:
    """Decide which agent to call next, or end the process."""
    history = state.messages

    response: AIMessage = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the supervisor. Based on the conversation so far, decide the next agent.\n"
                    "Agent 1:Focused on answering general knowledge questions."
                    "Agent 2:Focused on analysis and strategic advice."
                    "Only return JSON: {\"next_agent\": \"agent_1\" | \"agent_2\" | \"__end__\"}"
                ),
            },
            *history,
        ],
        response_format={"type": "json_object"},
    )
    print(response)
    parsed = json.loads(response.content)
    next_agent: str = parsed["next_agent"]
    return Command(goto=next_agent)

def agent_1(state: State) -> Command[Literal["supervisor"]]:
    """Focused on answering general knowledge questions."""
    history = state.messages
    reply: AIMessage = llm.invoke(
        [
            {"role": "system", "content": "You are Agent 1, specialized in general knowledge queries."},
            *history,
        ]
    )
    return Command(goto="supervisor", update={"messages": [reply]})

def agent_2(state: State) -> Command[Literal["supervisor"]]:
    """Focused on analysis and strategic advice."""
    history = state.messages
    reply: AIMessage = llm.invoke(
        [
            {"role": "system", "content": "You are Agent 2, specialized in analysis and strategic guidance."},
            *history,
        ]
    )
    return Command(goto="supervisor", update={"messages": [reply]})

# ── Build LangGraph workflow ────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("supervisor", supervisor)
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)

# Define entry point
builder.set_entry_point("supervisor")
# The supervisor will dynamically decide which agent to call next

graph = builder.compile(name="Supervisor-with-two-agents")
