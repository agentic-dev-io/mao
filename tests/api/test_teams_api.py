"""
Tests for the teams API endpoints.
"""

import pytest
import uuid


def test_create_team(api_test_client):
    """Test creating a team."""
    client, _ = api_test_client
    team_data = {"name": "Test Team", "description": "A test team"}

    response = client.post("/teams/", json=team_data)
    assert response.status_code == 201

    team = response.json()
    assert team["name"] == team_data["name"]
    assert team["description"] == team_data["description"]
    assert "id" in team
    assert "created_at" in team


@pytest.mark.skip(reason="Supervisor endpoints not implemented yet")
def test_create_and_get_supervisor(api_test_client):
    """Test creating and getting a supervisor."""
    client, _ = api_test_client

    # Create a team first
    team_data = {"name": "Team with Supervisor", "description": "For supervisor test"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create a supervisor
    supervisor_data = {
        "name": "Test Supervisor",
        "description": "A test supervisor",
        "team_id": team_id,
        "model": "gpt-4",
        "prompt": "You are a supervisor",
    }

    supervisor_response = client.post("/supervisors/", json=supervisor_data)
    assert supervisor_response.status_code == 201

    supervisor = supervisor_response.json()
    assert supervisor["name"] == supervisor_data["name"]
    assert supervisor["team_id"] == team_id
    assert "id" in supervisor

    # Get the supervisor
    get_response = client.get(f"/supervisors/{supervisor['id']}")
    assert get_response.status_code == 200

    get_supervisor = get_response.json()
    assert get_supervisor["id"] == supervisor["id"]
    assert get_supervisor["name"] == supervisor_data["name"]


@pytest.mark.skip(reason="Supervisor and member endpoints not implemented yet")
def test_team_with_supervisor_and_members(api_test_client):
    """Test creating a team with a supervisor and members."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Full Team", "description": "Team with supervisor and members"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create a supervisor
    supervisor_data = {
        "name": "Team Supervisor",
        "description": "The team supervisor",
        "team_id": team_id,
        "model": "gpt-4",
        "prompt": "You are the team supervisor",
    }

    supervisor_response = client.post("/supervisors/", json=supervisor_data)
    supervisor_id = supervisor_response.json()["id"]

    # Create agents
    agent1_data = {
        "name": "Team Member 1",
        "description": "First team member",
        "model": "gpt-3.5-turbo",
    }

    agent2_data = {
        "name": "Team Member 2",
        "description": "Second team member",
        "model": "gpt-3.5-turbo",
    }

    agent1_response = client.post("/agents/", json=agent1_data)
    agent2_response = client.post("/agents/", json=agent2_data)

    agent1_id = agent1_response.json()["id"]
    agent2_id = agent2_response.json()["id"]

    # Add agents to team
    member1_data = {
        "agent_id": agent1_id,
        "team_id": team_id,
        "role": "assistant",
    }

    member2_data = {
        "agent_id": agent2_id,
        "team_id": team_id,
        "role": "researcher",
    }

    member1_response = client.post("/team-members/", json=member1_data)
    member2_response = client.post("/team-members/", json=member2_data)

    assert member1_response.status_code == 201
    assert member2_response.status_code == 201

    # Get team details
    team_response = client.get(f"/teams/{team_id}")
    team = team_response.json()

    # Check team has correct supervisor and members
    assert team["supervisor_id"] == supervisor_id
    assert len(team["members"]) == 2
    member_agent_ids = [m["agent_id"] for m in team["members"]]
    assert agent1_id in member_agent_ids
    assert agent2_id in member_agent_ids


def test_team_update_and_delete(api_test_client):
    """Test updating and deleting a team."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Original Team", "description": "Original description"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Update the team
    update_data = {"name": "Updated Team", "description": "Updated description"}
    update_response = client.put(f"/teams/{team_id}", json=update_data)

    assert update_response.status_code == 200
    updated_team = update_response.json()
    assert updated_team["name"] == update_data["name"]
    assert updated_team["description"] == update_data["description"]

    # Delete the team
    delete_response = client.delete(f"/teams/{team_id}")
    assert delete_response.status_code == 204

    # Verify team is deleted
    get_response = client.get(f"/teams/{team_id}")
    assert get_response.status_code == 404


@pytest.mark.skip(reason="Member endpoints not implemented yet")
def test_team_member_update_and_remove(api_test_client):
    """Test updating and removing a team member."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Member Test Team", "description": "For member tests"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create an agent
    agent_data = {
        "name": "Test Agent",
        "description": "For member test",
        "model": "gpt-3.5-turbo",
    }
    agent_response = client.post("/agents/", json=agent_data)
    agent_id = agent_response.json()["id"]

    # Add agent to team
    member_data = {
        "agent_id": agent_id,
        "team_id": team_id,
        "role": "assistant",
    }
    member_response = client.post("/team-members/", json=member_data)
    member_id = member_response.json()["id"]

    # Update member role
    update_data = {"role": "researcher"}
    update_response = client.put(f"/team-members/{member_id}", json=update_data)

    assert update_response.status_code == 200
    updated_member = update_response.json()
    assert updated_member["role"] == update_data["role"]

    # Remove member from team
    delete_response = client.delete(f"/team-members/{member_id}")
    assert delete_response.status_code == 204

    # Verify member is removed
    team_response = client.get(f"/teams/{team_id}")
    team = team_response.json()
    assert len(team["members"]) == 0


@pytest.mark.skip(reason="Supervisor endpoints not implemented yet")
def test_supervisor_lifecycle(api_test_client):
    """Test the full lifecycle of a supervisor."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Supervisor Team", "description": "For supervisor lifecycle"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create a supervisor
    supervisor_data = {
        "name": "Lifecycle Supervisor",
        "description": "Testing lifecycle",
        "team_id": team_id,
        "model": "gpt-4",
        "prompt": "Initial prompt",
    }

    supervisor_response = client.post("/supervisors/", json=supervisor_data)
    supervisor_id = supervisor_response.json()["id"]

    # Update the supervisor
    update_data = {
        "name": "Updated Supervisor",
        "prompt": "Updated prompt",
    }
    update_response = client.put(f"/supervisors/{supervisor_id}", json=update_data)

    assert update_response.status_code == 200
    updated_supervisor = update_response.json()
    assert updated_supervisor["name"] == update_data["name"]
    assert updated_supervisor["prompt"] == update_data["prompt"]

    # Delete the supervisor
    delete_response = client.delete(f"/supervisors/{supervisor_id}")
    assert delete_response.status_code == 204

    # Verify supervisor is deleted
    get_response = client.get(f"/supervisors/{supervisor_id}")
    assert get_response.status_code == 404

    # Verify team no longer has a supervisor
    team_response = client.get(f"/teams/{team_id}")
    team = team_response.json()
    assert team.get("supervisor_id") is None


def test_get_team_not_found(api_test_client):
    """Test getting a non-existent team."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.get(f"/teams/{random_id}")
    assert response.status_code == 404


def test_update_team_not_found(api_test_client):
    """Test updating a non-existent team."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    update_data = {"name": "Updated Team", "description": "Updated description"}
    response = client.put(f"/teams/{random_id}", json=update_data)
    assert response.status_code == 404


def test_delete_team_not_found(api_test_client):
    """Test deleting a non-existent team."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.delete(f"/teams/{random_id}")
    assert response.status_code == 404


def test_get_supervisor_not_found(api_test_client):
    """Test getting a non-existent supervisor."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.get(f"/supervisors/{random_id}")
    assert response.status_code == 404


def test_update_supervisor_not_found(api_test_client):
    """Test updating a non-existent supervisor."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    update_data = {"name": "Updated Supervisor", "prompt": "Updated prompt"}
    response = client.put(f"/supervisors/{random_id}", json=update_data)
    assert response.status_code == 404


def test_delete_supervisor_not_found(api_test_client):
    """Test deleting a non-existent supervisor."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.delete(f"/supervisors/{random_id}")
    assert response.status_code == 404


def test_list_teams(api_test_client):
    """Test listing all teams."""
    client, _ = api_test_client

    # Create some teams
    team1_data = {"name": "Team 1", "description": "First team"}
    team2_data = {"name": "Team 2", "description": "Second team"}

    client.post("/teams/", json=team1_data)
    client.post("/teams/", json=team2_data)

    # List all teams
    response = client.get("/teams/")
    assert response.status_code == 200

    teams = response.json()
    assert isinstance(teams, list)
    assert len(teams) >= 2
    team_names = [t["name"] for t in teams]
    assert "Team 1" in team_names
    assert "Team 2" in team_names


@pytest.mark.skip(reason="Supervisor endpoints not implemented yet")
def test_list_supervisors(api_test_client):
    """Test listing all supervisors."""
    client, _ = api_test_client

    # Create a team
    team_data = {
        "name": "Supervisor List Team",
        "description": "For listing supervisors",
    }
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create some supervisors
    supervisor1_data = {
        "name": "Supervisor 1",
        "description": "First supervisor",
        "team_id": team_id,
        "model": "gpt-4",
        "prompt": "Prompt 1",
    }

    supervisor2_data = {
        "name": "Supervisor 2",
        "description": "Second supervisor",
        "team_id": team_id,
        "model": "gpt-4",
        "prompt": "Prompt 2",
    }

    client.post("/supervisors/", json=supervisor1_data)
    client.post("/supervisors/", json=supervisor2_data)

    # List all supervisors
    response = client.get("/supervisors/")
    assert response.status_code == 200

    supervisors = response.json()
    assert isinstance(supervisors, list)
    assert len(supervisors) >= 2
    supervisor_names = [s["name"] for s in supervisors]
    assert "Supervisor 1" in supervisor_names
    assert "Supervisor 2" in supervisor_names


@pytest.mark.skip(reason="Member endpoints not implemented yet")
def test_get_team_members(api_test_client):
    """Test getting members of a specific team."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Member List Team", "description": "For listing members"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create agents
    agent1_data = {
        "name": "Member Agent 1",
        "description": "First member agent",
        "model": "gpt-3.5-turbo",
    }
    agent2_data = {
        "name": "Member Agent 2",
        "description": "Second member agent",
        "model": "gpt-3.5-turbo",
    }

    agent1_response = client.post("/agents/", json=agent1_data)
    agent2_response = client.post("/agents/", json=agent2_data)

    agent1_id = agent1_response.json()["id"]
    agent2_id = agent2_response.json()["id"]

    # Add agents to team
    member1_data = {
        "agent_id": agent1_id,
        "team_id": team_id,
        "role": "assistant",
    }

    member2_data = {
        "agent_id": agent2_id,
        "team_id": team_id,
        "role": "researcher",
    }

    client.post("/team-members/", json=member1_data)
    client.post("/team-members/", json=member2_data)

    # Get team members
    response = client.get(f"/teams/{team_id}/members")
    assert response.status_code == 200

    members = response.json()
    assert isinstance(members, list)
    assert len(members) == 2
    member_agent_ids = [m["agent_id"] for m in members]
    assert agent1_id in member_agent_ids
    assert agent2_id in member_agent_ids
