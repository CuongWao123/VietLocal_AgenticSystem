"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations


from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

from agent.Configuration import Configuration
from agent.State import State  


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Call GPT-4o-mini model with runtime config."""
    
    configuration = config["configurable"]
    model_name = configuration.get("model_name", "gpt-4o-mini")

    # Init LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
    )

    # Format input message
    prompt = "What is the capital of France?"

    # Call model asynchronously
    response = await llm.ainvoke(prompt)

    return {
        "messages": response.content
    }

# Define the graph
graph_builder = StateGraph(State, config_schema=Configuration)

graph_builder.add_node("call_model", call_model)


graph_builder.set_entry_point("call_model")


graph = graph_builder.compile(name="New Graph")
