"""
MAO Agents: Modern agent framework with KnowledgeTree, ExperienceTree, and LangChain integration.
"""

__version__ = "1.0.0"

from .agents import create_agent, Agent, Supervisor

from .storage import (
    KnowledgeTree,
    ExperienceTree,
    VectorStoreBase,
    VectorStoreError,
    SearchResult,
)

from .mcp import MCPClient, ToolConfig, ServerConfig

__all__ = [
    "create_agent",
    "Agent",
    "Supervisor",
    "KnowledgeTree",
    "ExperienceTree",
    "VectorStoreBase",
    "VectorStoreError",
    "SearchResult",
    "MCPClient",
    "ToolConfig",
    "ServerConfig",
    "__version__",
]
