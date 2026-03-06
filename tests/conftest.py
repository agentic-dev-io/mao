"""
Pytest fixtures for mao tests.
"""

import os
import pytest
import asyncio
import logging
import httpx
import socket
import subprocess
import time
import threading
import uuid
import uvicorn
from dotenv import load_dotenv
from mao.mcp import MCPClient
from fastapi.testclient import TestClient
from mao.api.api import MCPAgentsAPI
from mao.api.db import ConfigDB

load_dotenv()

try:
    from pytest_asyncio import fixture as asyncio_fixture
except ImportError:
    asyncio_fixture = pytest.fixture  # type: ignore


PREFERRED_HITL_CLOUD_MODELS = (
    "gpt-oss:20b-cloud",
    "qwen3.5:cloud",
    "glm-5:cloud",
)


@asyncio_fixture(scope="function")
async def mcp_client():
    client = None
    try:
        client = MCPClient()
        logging.info(
            f"MCPClient for tests with {len(client.list_servers())} configured servers"
        )
        yield client
    finally:
        if client is not None:
            del client


@pytest.fixture(scope="function")
def api_test_client():
    base_tmp_dir = os.path.join(os.getcwd(), ".test_tmp")
    os.makedirs(base_tmp_dir, exist_ok=True)

    previous_mcp_db_path = os.environ.get("MCP_DB_PATH")
    previous_vector_db_path = os.environ.get("VECTOR_DB_PATH")
    previous_checkpoint_db_path = os.environ.get("MAO_CHECKPOINT_DB_PATH")
    test_run_id = uuid.uuid4().hex
    os.environ["MCP_DB_PATH"] = os.path.join(base_tmp_dir, f"api_{test_run_id}.duckdb")
    os.environ["VECTOR_DB_PATH"] = os.path.join(
        base_tmp_dir, f"api_vectors_{test_run_id}.duckdb"
    )
    os.environ["MAO_CHECKPOINT_DB_PATH"] = os.path.join(
        base_tmp_dir, f"api_checkpoints_{test_run_id}.duckdb"
    )
    test_api = MCPAgentsAPI(
        db_path=":memory:", title="Test MCP Agents API", version="test"
    )
    client = TestClient(test_api)
    try:
        yield client, test_api
    finally:
        asyncio.run(ConfigDB.cleanup())
        if previous_checkpoint_db_path is None:
            os.environ.pop("MAO_CHECKPOINT_DB_PATH", None)
        else:
            os.environ["MAO_CHECKPOINT_DB_PATH"] = previous_checkpoint_db_path
        if previous_mcp_db_path is None:
            os.environ.pop("MCP_DB_PATH", None)
        else:
            os.environ["MCP_DB_PATH"] = previous_mcp_db_path
        if previous_vector_db_path is None:
            os.environ.pop("VECTOR_DB_PATH", None)
        else:
            os.environ["VECTOR_DB_PATH"] = previous_vector_db_path


@pytest.fixture(scope="session")
def real_tool_calling_cloud_model():
    requested_model = os.environ.get("TEST_HITL_MODEL")
    available_models = _list_ollama_models()

    if requested_model:
        if requested_model not in available_models:
            pytest.skip(
                f"Requested TEST_HITL_MODEL '{requested_model}' is not available in ollama list"
            )
        return requested_model

    for model_name in PREFERRED_HITL_CLOUD_MODELS:
        if model_name in available_models:
            return model_name

    pytest.skip(
        "No verified tool-calling cloud model available. "
        f"Checked: {', '.join(PREFERRED_HITL_CLOUD_MODELS)}"
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class UvicornTestServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @property
    def is_running(self):
        return self.started


@pytest.fixture(scope="session")
def live_api_server():
    api = MCPAgentsAPI(
        db_path="test_live_api.duckdb", title="Live Test MCP Agents API", version="test"
    )

    port = find_free_port()
    host = "127.0.0.1"

    config = uvicorn.Config(app=api, host=host, port=port, log_level="error")
    server = UvicornTestServer(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    startup_timeout = 5.0
    start_time = time.time()
    while not server.is_running and time.time() - start_time < startup_timeout:
        time.sleep(0.1)

    if not server.is_running:
        raise RuntimeError("Failed to start API server within timeout period")

    base_url = f"http://{host}:{port}"

    client = httpx.Client(base_url=base_url, timeout=30.0)

    try:
        response = client.get("/health")
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to API server: {e}")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


def pytest_sessionfinish(session, exitstatus):
    if os.name == "nt":
        try:
            asyncio.run(asyncio.sleep(1))
        except Exception:
            time.sleep(1)

    test_db_path = "test_live_api.duckdb"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception as e:
            logging.warning(f"Could not remove test database: {e}")


def _list_ollama_models() -> set[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"ollama list not available for HITL tests: {exc}")

    models: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.add(parts[0])
    return models
