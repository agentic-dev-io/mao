"""
Tests for the MCP API endpoints.
Test CRUD operations on servers, tools, and related functionality.
"""

import random
import string
import httpx


def generate_unique_name(prefix="Test"):
    """Generate a unique name with a random suffix."""
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{random_suffix}"


def test_create_server(api_test_client):
    """Test creating a new server."""
    client, _ = api_test_client

    # Test data for server creation with unique name
    server_name = generate_unique_name("TestServer")
    server_data = {
        "name": server_name,
        "transport": "stdio",
        "enabled": True,
        "command": "python",
        "args": ["-m", "mcp_agents.server"],
        "env_vars": {"TEST_ENV": "test_value"},
    }

    response = client.post("/mcp/servers", json=server_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == server_data["name"]
    assert data["transport"] == server_data["transport"]
    assert data["enabled"] == server_data["enabled"]
    assert data["command"] == server_data["command"]
    assert data["args"] == server_data["args"]
    assert data["env_vars"] == server_data["env_vars"]
    assert "id" in data
    assert data["id"].startswith("server_")


def test_list_servers(api_test_client):
    """Test listing all servers."""
    client, _ = api_test_client

    # Create a server first with a unique name
    server_name = generate_unique_name("ServerList")
    server_data = {
        "name": server_name,
        "transport": "websocket",
        "enabled": True,
        "url": "ws://localhost:8080",
    }

    create_response = client.post("/mcp/servers", json=server_data)
    assert create_response.status_code == 201

    # Get the list of all servers
    list_response = client.get("/mcp/servers")
    assert list_response.status_code == 200

    servers = list_response.json()
    assert isinstance(servers, list)
    assert len(servers) >= 1

    # Find the created server in the list
    found = False
    for server in servers:
        if server["name"] == server_data["name"]:
            found = True
            break

    assert found, "Created server was not found in the list"


def test_create_tool(api_test_client):
    """Test creating a new tool."""
    client, _ = api_test_client

    # Create a server first for the tool with a unique name
    server_name = generate_unique_name("ServerForTool")
    server_data = {"name": server_name, "transport": "stdio", "enabled": True}

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201
    server_id = server_response.json()["id"]

    # Test data for tool creation
    tool_data = {
        "name": "Test Tool",
        "enabled": True,
        "server_id": server_id,
        "description": "A test tool for API testing",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    }

    response = client.post("/mcp/tools", json=tool_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == tool_data["name"]
    assert data["enabled"] == tool_data["enabled"]
    assert data["server_id"] == server_id
    assert data["description"] == tool_data["description"]
    assert "id" in data
    assert data["id"].startswith("tool_")


def test_list_tools(api_test_client):
    """Test listing all tools."""
    client, _ = api_test_client

    # Create a server first with a unique name
    server_name = generate_unique_name("ToolListServer")
    server_data = {"name": server_name, "transport": "stdio", "enabled": True}

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201
    server_id = server_response.json()["id"]

    # Create a tool
    tool_data = {"name": "Test Tool for List", "enabled": True, "server_id": server_id}

    create_response = client.post("/mcp/tools", json=tool_data)
    assert create_response.status_code == 201

    # Get the list of all tools
    list_response = client.get("/mcp/tools")
    assert list_response.status_code == 200

    tools = list_response.json()
    assert isinstance(tools, list)
    assert len(tools) >= 1

    # Find the created tool in the list
    found = False
    for tool in tools:
        if tool["name"] == tool_data["name"]:
            found = True
            break

    assert found, "Created tool was not found in the list"

    # Test filtering by server ID
    filtered_response = client.get(f"/mcp/tools?server_id={server_id}")
    assert filtered_response.status_code == 200

    filtered_tools = filtered_response.json()
    assert len(filtered_tools) >= 1
    assert all(tool["server_id"] == server_id for tool in filtered_tools)


def test_get_mcp_config(api_test_client):
    """Test getting MCP configuration."""
    client, _ = api_test_client

    # Create a server for the configuration with a unique name
    server_name = generate_unique_name("ConfigServer")
    server_data = {
        "name": server_name,
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "server"],
        "enabled": True,
    }

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201

    # Test the configuration endpoint
    config_response = client.get("/mcp/config")
    assert config_response.status_code == 200

    config = config_response.json()
    assert "mcpServers" in config
    assert isinstance(config["mcpServers"], dict)


def test_assign_tool_to_agent(api_test_client):
    """Test assigning a tool to an agent."""
    client, _ = api_test_client

    # Create an agent
    agent_data = {
        "name": "Tool Assignment Test Agent",
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
    }

    agent_response = client.post("/agents", json=agent_data)
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]

    # Create a server with a unique name
    server_name = generate_unique_name("ToolAssignmentServer")
    server_data = {"name": server_name, "transport": "stdio", "enabled": True}

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201
    server_id = server_response.json()["id"]

    # Create a tool
    tool_data = {
        "name": "Tool for Assignment Test",
        "enabled": True,
        "server_id": server_id,
    }

    tool_response = client.post("/mcp/tools", json=tool_data)
    assert tool_response.status_code == 201
    tool_id = tool_response.json()["id"]

    # Assign the tool to the agent
    assignment_data = {"enabled": True}

    assignment_response = client.post(
        f"/mcp/tools/agent/{agent_id}/tool/{tool_id}", json=assignment_data
    )
    assert assignment_response.status_code == 204

    # Check if the tool is in the agent's tool list
    agent_tools_response = client.get(f"/agents/{agent_id}/tools")
    assert agent_tools_response.status_code == 200

    agent_tools = agent_tools_response.json()
    assert isinstance(agent_tools, list)
    assert len(agent_tools) >= 1

    # Find the assigned tool in the list
    found = False
    for tool in agent_tools:
        if tool["id"] == tool_id:
            found = True
            assert tool["enabled"] == assignment_data["enabled"]
            break

    assert found, "Assigned tool was not found in the agent's tool list"

    # Remove the tool from the agent
    remove_response = client.delete(f"/mcp/tools/agent/{agent_id}/tool/{tool_id}")
    assert remove_response.status_code == 204

    # Check that the tool is no longer in the agent's tool list
    agent_tools_response_after = client.get(f"/agents/{agent_id}/tools")
    assert agent_tools_response_after.status_code == 200

    agent_tools_after = agent_tools_response_after.json()
    not_found = True
    for tool in agent_tools_after:
        if tool["id"] == tool_id:
            not_found = False
            break

    assert not_found, "Tool was not successfully removed from the agent"


def test_server_tool_lifecycle_with_live_server(live_api_server):
    """Test complete server and tool lifecycle with a live server."""
    with httpx.Client(base_url=live_api_server, timeout=10.0) as client:
        # 1. Create a server
        server_name = generate_unique_name("LiveServer")
        server_data = {
            "name": server_name,
            "transport": "websocket",
            "enabled": True,
            "url": "ws://localhost:9000/ws",
            "headers": {"X-API-KEY": "test-key"},
        }

        server_response = client.post("/mcp/servers", json=server_data)
        assert server_response.status_code == 201
        server = server_response.json()
        server_id = server["id"]

        # 2. Create a tool for the server
        tool_name = generate_unique_name("LiveTool")
        tool_data = {
            "name": tool_name,
            "enabled": True,
            "server_id": server_id,
            "description": "A test tool for live server testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Tool input"}
                },
            },
        }

        tool_response = client.post("/mcp/tools", json=tool_data)
        assert tool_response.status_code == 201
        tool = tool_response.json()
        tool_id = tool["id"]

        # 3. Create an agent
        agent_data = {
            "name": generate_unique_name("LiveAgent"),
            "provider": "openai",
            "model_name": "gpt-4",
        }

        agent_response = client.post("/agents", json=agent_data)
        assert agent_response.status_code == 201
        agent = agent_response.json()
        agent_id = agent["id"]

        # 4. Assign the tool to the agent
        assignment_data = {"enabled": True}
        assignment_response = client.post(
            f"/mcp/tools/agent/{agent_id}/tool/{tool_id}", json=assignment_data
        )
        assert assignment_response.status_code == 204

        # 5. Get the agent's tool list
        agent_tools_response = client.get(f"/agents/{agent_id}/tools")
        assert agent_tools_response.status_code == 200
        tools = agent_tools_response.json()
        assert any(t["id"] == tool_id for t in tools)

        # 6. Get the MCP configuration
        config_response = client.get("/mcp/config")
        assert config_response.status_code == 200
        config = config_response.json()
        assert "mcpServers" in config
        assert server_name in config["mcpServers"]

        # 7. Clean up the resources
        # Remove tool assignment
        remove_tool_response = client.delete(
            f"/mcp/tools/agent/{agent_id}/tool/{tool_id}"
        )
        assert remove_tool_response.status_code == 204

        # Delete tool
        delete_tool_response = client.delete(f"/mcp/tools/{tool_id}")
        assert delete_tool_response.status_code == 204

        # Delete server
        delete_server_response = client.delete(f"/mcp/servers/{server_id}")
        assert delete_server_response.status_code == 204

        # Delete agent
        delete_agent_response = client.delete(f"/agents/{agent_id}")
        assert delete_agent_response.status_code == 204
