from dataclasses import dataclass, field
from typing import Annotated, Sequence , Optional
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


@dataclass
class State:
    """
    Input state for the agent.
    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """


    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    next_agent: Optional[str] = None
    is_last_step: IsLastStep = field(default=False)
