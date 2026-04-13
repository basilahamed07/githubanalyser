from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Input
    user_query: str
    repo_owner: str
    repo_name: str

    # Planning & Control
    intent: str
    domain: list
    plan: list
    current_step: int
    next_node: str
    loop_count: int
    is_complete: bool
    needs_clarification: bool

    # Execution
    messages: Annotated[list, add_messages]
    tool_calls_made: list
    intermediate_results: dict

    # Output
    final_answer: str
    error: Optional[str]