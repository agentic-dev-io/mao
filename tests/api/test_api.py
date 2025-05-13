"""
Tests for the MCP Agents API.
Basic API endpoint functionality tests.
"""


def test_health_endpoint(api_test_client):
    """Test that the health endpoint returns status ok."""
    client, _ = api_test_client
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_root_endpoint(api_test_client):
    """Test that the root endpoint returns API information."""
    client, _ = api_test_client
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert "version" in data
    assert "endpoints" in data
    assert isinstance(data["endpoints"], dict)
