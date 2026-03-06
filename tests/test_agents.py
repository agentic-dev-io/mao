"""
Tests for Agent and Supervisor classes.
"""

import pytest
import uuid
import os
import logging

from langchain_core.messages import AIMessage, HumanMessage

from mao.agents import create_agent, Supervisor, load_mcp_tools

# Configuration for tests — defaults to Ollama (local/cloud)
TEST_LLM_PROVIDER = os.environ.get("TEST_LLM_PROVIDER", "ollama")
TEST_LLM_MODEL = os.environ.get("TEST_LLM_MODEL", "gemma3:4b-cloud")


@pytest.fixture(autouse=True)
def isolated_vector_db():
    previous_vector_db_path = os.environ.get("VECTOR_DB_PATH")
    base_tmp_dir = os.path.join(os.getcwd(), ".test_tmp")
    os.makedirs(base_tmp_dir, exist_ok=True)
    os.environ["VECTOR_DB_PATH"] = os.path.join(
        base_tmp_dir, f"agent_vectors_{uuid.uuid4().hex}.duckdb"
    )
    try:
        yield
    finally:
        if previous_vector_db_path is None:
            os.environ.pop("VECTOR_DB_PATH", None)
        else:
            os.environ["VECTOR_DB_PATH"] = previous_vector_db_path


@pytest.mark.asyncio
async def test_create_agent_factory():
    """Test the agent factory function with basic parameters."""
    agent_name = f"test_agent_{uuid.uuid4().hex[:8]}"
    agent_app = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name=agent_name,
        system_prompt="You are a test agent.",
    )

    # Verify basic agent properties
    assert agent_app is not None, "Agent app should be created"
    assert hasattr(agent_app, "ainvoke"), "Agent should have ainvoke method"
    assert agent_app.name == agent_name, "Agent name should be set correctly"

    # Basic functionality test
    thread_id = f"test_thread_{uuid.uuid4()}"
    response = await agent_app.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Verify response structure
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    last_message = response["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content, "Agent response should not be empty"


@pytest.mark.asyncio
async def test_agent_multi_turn_conversation():
    """Test that agent maintains coherence across multiple turns via thread_id."""
    agent_app = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name="conversation_agent_test",
        system_prompt="You remember conversations. Always answer concisely.",
    )

    thread_id = f"conversation_thread_{uuid.uuid4()}"

    # First turn: establish a fact
    await agent_app.ainvoke(
        {"messages": [HumanMessage(content="My name is Wolfgang. Remember that.")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Second turn: ask about the fact from the same thread
    follow_up_response = await agent_app.ainvoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    last_message = follow_up_response["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert "Wolfgang" in last_message.content, (
        "Agent should recall the name from the earlier turn in the same thread"
    )


@pytest.mark.asyncio
async def test_supervisor_basic():
    """Test Supervisor with a simple worker agent."""
    # Create worker agent
    worker_agent = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name="worker_agent",
        system_prompt="You are a helpful worker. Answer questions directly and concisely.",
    )

    # Create supervisor
    supervisor = Supervisor(
        agents=[worker_agent],
        supervisor_provider=TEST_LLM_PROVIDER,
        supervisor_model_name=TEST_LLM_MODEL,
        supervisor_system_prompt=(
            "You are a supervisor. Delegate the user's question to worker_agent."
        ),
    )

    # Initialize the supervisor
    await supervisor.init_supervisor()
    assert supervisor.app is not None, "Supervisor should be initialized"

    # Test supervisor delegation
    thread_id = f"supervisor_thread_{uuid.uuid4()}"
    response = await supervisor.invoke(
        messages=[HumanMessage(content="What is AI?")], thread_id=thread_id
    )

    # Verify response
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0

    # Find any AI response (worker or supervisor)
    ai_response = None
    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            ai_response = msg.content
            # Prefer worker response if available
            if getattr(msg, "name", None) == "worker_agent":
                break

    assert ai_response is not None, "Supervisor should have produced a response"
    assert len(ai_response) > 0, "Response should not be empty"


@pytest.mark.asyncio
async def test_load_mcp_tools_function(mcp_client):
    """Test the load_mcp_tools helper function with different inputs."""
    # Test with real tools from an MCPClient (no context manager needed)
    try:
        real_tools = await mcp_client.get_tools()
    except Exception as e:
        pytest.skip(f"MCP server not reachable: {e}")

    loaded_tools = await load_mcp_tools(real_tools)
    assert (
        loaded_tools == real_tools
    ), "Should return the same list when input is a list"

    # Test with None
    empty_tools = await load_mcp_tools(None)
    assert empty_tools == [], "Should return empty list when input is None"

    # Test with invalid type
    string_input = "not a valid tools input"
    invalid_tools = await load_mcp_tools(string_input)
    assert invalid_tools == [], "Should return empty list when input is invalid type"


@pytest.mark.asyncio
async def test_agent_with_mcp_tools(mcp_client):
    """Test creating an agent with MCP tools."""
    # Check if MCP client works
    servers = mcp_client.list_servers()
    if not servers:
        pytest.skip("No MCP servers configured")

    active_servers = mcp_client.list_active_servers()
    if not active_servers:
        pytest.skip("No active MCP servers found")

    # Create agent with MCP tools
    agent_app = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name="mcp_tools_agent",
        tools=mcp_client,
        system_prompt="You are a helpful assistant with access to tools. Use the appropriate tool when needed.",
    )

    # Verify agent was created
    assert agent_app is not None, "Agent app should be created"
    assert hasattr(agent_app, "ainvoke"), "Agent should have ainvoke method"

    # Test with a request that should use a tool
    thread_id = f"mcp_tools_thread_{uuid.uuid4()}"
    test_query = "What is the current time in Berlin, Germany?"

    try:
        response = await agent_app.ainvoke(
            {"messages": [HumanMessage(content=test_query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Verify response
        assert response is not None
        assert "messages" in response
        assert len(response["messages"]) > 0

        # Log the response but don't enforce specific content
        last_message = response["messages"][-1]
        logging.info(f"Response to '{test_query}': {last_message.content}")
    except Exception as e:
        logging.error(f"Error testing agent with MCP tools: {e}")
        raise
