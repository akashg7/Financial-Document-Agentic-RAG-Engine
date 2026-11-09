r"""
Reference Solution for Hand-Write Piece #3: LangGraph StateGraph Definition
File: app/agent/graph.py

Topology:
        +-------+
        | START |
        +-------+
            |
            v
      +------------+
      |  retrieve  |<--------------------+
      +------------+                     |
            |                            |
            v                            |
      +------------+                     |
      |  generate  |                     |
      +------------+                     |
            |                            |
     [route_decision]                    |
      /     |      \                     |
"generate" "rewrite" "abstain"          |
    /       |        \                   |
   v        v         v                  |
+-----+ +---------+ +---------+          |
| END | | rewrite | | abstain |          |
+-----+ +---------+ +---------+          |
             |            |              |
             +------------+              |
                          |              |
                          v              |
                       +-----+           |
                       | END |           |
                       +-----+           |
"""

from typing import Any
from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes import (
    retrieve_node,
    generate_node,
    rewrite_node,
    abstain_node,
    route_decision,
)


def build_agent_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph for the agentic RAG loop.
    """
    # 1. Initialize StateGraph with our typed state
    builder = StateGraph(AgentState)

    # 2. Register all workflow nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("abstain", abstain_node)

    # 3. Connect entry point and sequential edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")

    # 4. Attach conditional routing logic from generate
    builder.add_conditional_edges(
        "generate",
        route_decision,
        {
            "generate": END,
            "rewrite": "rewrite",
            "abstain": "abstain",
        },
    )

    # 5. Connect rewrite loop back to retrieve
    builder.add_edge("rewrite", "retrieve")

    # 6. Connect terminal abstain node to END
    builder.add_edge("abstain", END)

    # 7. Compile state graph into runnable application
    return builder.compile()
