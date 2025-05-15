"""
Tests for MCPClient.
Production-ready, async, and robust implementation.
"""

import pytest
import asyncio
import logging

# Import pytest_asyncio if available
try:
    import pytest_asyncio

    ASYNCIO_FIXTURE = pytest_asyncio.fixture
except ImportError:
    ASYNCIO_FIXTURE = pytest.fixture  # type: ignore


@pytest.mark.asyncio
async def test_mcp_client_tool_states(mcp_client):
    """Test MCPClient tool enablement logic."""
    # Set initial tool states
    tools_to_test = ["test_tool1", "test_tool2"]

    # Enable tools and verify
    for tool in tools_to_test:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, True)
        assert await asyncio.to_thread(
            mcp_client.is_tool_enabled, tool
        ), f"Tool {tool} should be enabled"

    # Disable tools and verify
    for tool in tools_to_test:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, False)
        assert not await asyncio.to_thread(
            mcp_client.is_tool_enabled, tool
        ), f"Tool {tool} should be disabled"

    # Test with non-existent tool
    assert not await asyncio.to_thread(
        mcp_client.is_tool_enabled, "non_existent_tool"
    ), "Non-existent tool should report as disabled"


@pytest.mark.asyncio
async def test_mcp_client_reload(mcp_client):
    """Test configuration reloading."""
    # Get initial state
    initial_servers = await asyncio.to_thread(mcp_client.list_servers)

    # Set some tool states before reload
    test_tools = ["tool_before_reload1", "tool_before_reload2"]
    for tool in test_tools:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, True)

    # Reload configuration
    await asyncio.to_thread(mcp_client.reload)

    # Verify servers maintained after reload
    current_servers = await asyncio.to_thread(mcp_client.list_servers)
    assert (
        current_servers == initial_servers
    ), "Server list should be maintained after reload"

    # Verify tool states maintained
    for tool in test_tools:
        assert await asyncio.to_thread(
            mcp_client.is_tool_enabled, tool
        ), f"Tool {tool} should remain enabled after reload"


@pytest.mark.asyncio
async def test_mcp_client_get_tools(mcp_client):
    """Test getting tools from MCP servers."""
    # Test getting tools in a context
    async with mcp_client.session() as client_in_context:
        # get_tools is a synchronous method
        tools = client_in_context.get_tools()
        assert tools is not None, "No tools found from MCP servers"
        assert isinstance(tools, list), "get_tools should return a list"

        # Log detailed information about found tools
        logging.info(f"Found: {len(tools)} tools from MCP servers")

        for i, tool in enumerate(tools):
            tool_info = {
                "name": getattr(tool, "name", "Unknown"),
                "description": getattr(tool, "description", "No description"),
                "type": type(tool).__name__,
            }
            # Additional properties if available
            if hasattr(tool, "args_schema"):
                tool_info["args_schema"] = str(tool.args_schema)

            logging.info(f"Tool {i+1}: {tool_info}")

    # Test tool activation for specific tools if available
    if tools:
        test_tool = tools[0].name
        await asyncio.to_thread(mcp_client.set_tool_enabled, test_tool, True)
        assert await asyncio.to_thread(mcp_client.is_tool_enabled, test_tool)
        await asyncio.to_thread(mcp_client.set_tool_enabled, test_tool, False)
        assert not await asyncio.to_thread(mcp_client.is_tool_enabled, test_tool)
