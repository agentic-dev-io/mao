"""Tests for the persistent DuckDB-backed LangGraph checkpointer."""

import os
import uuid
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mao.checkpoint import DuckDBSaver


def test_duckdb_saver_persists_graph_state():
    base_tmp_dir = Path(os.getcwd()) / ".test_tmp"
    base_tmp_dir.mkdir(exist_ok=True)
    checkpoint_path = base_tmp_dir / f"checkpoint_test_{uuid.uuid4().hex}.duckdb"
    saver = DuckDBSaver(str(checkpoint_path))

    def increment(state: dict) -> dict:
        return {"count": state.get("count", 0) + 1}

    builder = StateGraph(dict)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    graph = builder.compile(checkpointer=saver)
    thread_id = f"checkpoint-thread-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    first = graph.invoke({"count": 0}, config=config)
    second = graph.invoke(None, config=config)

    assert first["count"] == 1
    assert second["count"] == 1

    latest = saver.get_tuple(config)
    assert latest is not None
    assert latest.config["configurable"]["thread_id"] == thread_id
    assert os.path.exists(checkpoint_path)
