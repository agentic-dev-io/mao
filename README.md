# MAO - MCP Agent Orchestra

<div align="center">
  <p>
    <a href="https://github.com/tiangolo/fastapi"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"></a>
    <a href="https://github.com/qdrant/qdrant"><img src="https://img.shields.io/badge/Qdrant-FF4582?style=for-the-badge&logo=qdrant" alt="Qdrant"></a>
    <a href="https://github.com/duckdb/duckdb"><img src="https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb" alt="DuckDB"></a>
    <a href="https://github.com/langchain-ai/langchain"><img src="https://img.shields.io/badge/LangChain-2C39BD?style=for-the-badge&logo=langchain" alt="LangChain"></a>
  </p>
  <p>
    <a href="https://github.com/anthropics/anthropic-sdk-python"><img src="https://img.shields.io/badge/Anthropic-0B0D10?style=for-the-badge&logo=anthropic" alt="Anthropic"></a>
    <a href="https://github.com/openai/openai-python"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai" alt="OpenAI"></a>
    <a href="https://github.com/ollama/ollama"><img src="https://img.shields.io/badge/Ollama-121212?style=for-the-badge&logo=ollama" alt="Ollama"></a>
    <a href="https://github.com/modelcontextprotocol/modelcontextprotocol"><img src="https://img.shields.io/badge/MCP-6A0DAD?style=for-the-badge&logo=mcp" alt="MCP"></a>
  </p>
</div>

A modern framework for orchestrating AI agents using the Model Context Protocol (MCP).

## Overview

MAO (MCP Agent Orchestra) is a comprehensive infrastructure for creating, managing, and orchestrating AI agents. It provides a robust platform that integrates various Large Language Models (LLMs) with tools through the Model Context Protocol, enabling sophisticated agent interactions and knowledge management.

## Key Features

- **Flexible Agent Framework**: Create agents powered by various LLM providers ([OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Ollama](https://ollama.com/))
- **Team Orchestration**: Organize agents into teams with supervisor coordination
- **Knowledge Management**: Store and retrieve agent knowledge and experiences using vector databases
- **Tool Integration**: Dynamically load and manage tools via the [MCP protocol](https://modelcontextprotocol.io/)
- **REST API**: Complete management interface for all system resources

## Architecture

MAO consists of several core components:

1. **Agent Framework** (`agents.py`):
   - LLM integration with OpenAI, Anthropic, and Ollama
   - Advanced features like streaming, tool integration, and memory
   - Support for supervisor agents to coordinate teams

2. **MCP Integration** (`mcp.py`):
   - Client for the Model Context Protocol
   - Management of connections to various MCP servers
   - Dynamic loading and management of tools

3. **Storage System** (`storage.py`):
   - [Qdrant](https://qdrant.tech/) vector database for semantic search
   - KnowledgeTree and ExperienceTree for agent knowledge and experience
   - Fully asynchronous API for database operations

4. **REST API** (`api/` folder):
   - [FastAPI](https://fastapi.tiangolo.com/)-based REST interface for managing agents, teams, and tools
   - CRUD operations for all resources
   - [DuckDB](https://duckdb.org/) for lightweight configuration storage

5. **Configuration Management** (`db.py`):
   - Configuration management for agents, teams, servers, and tools
   - Asynchronous database interface
   - Singleton pattern for database connections

## Technical Details

MAO follows modern Python best practices:
- Complete typing with Type Hints
- Asynchronous programming with asyncio
- Error handling and retry logic
- Modular design with clear responsibilities

## Getting Started

1. Install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Run the API server:
   ```bash
   uvicorn src.mao.api.api:api --reload
   ```

4. Access the API documentation at `http://localhost:8000/docs`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
