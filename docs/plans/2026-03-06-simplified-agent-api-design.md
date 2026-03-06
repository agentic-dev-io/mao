# Simplified Agent API — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the agent API: trees internal, migrate to LangChain v1 `create_agent` + `init_chat_model`, remove deprecated `create_react_agent`.

**Architecture:** `create_agent()` takes only what the developer cares about (provider, model, name, prompt, tools). Internally uses `langchain.agents.create_agent` (v1) and `init_chat_model`. Trees created automatically from agent_name. No more manual LLM client factory.

**Tech Stack:** Python, LangChain v1, LangGraph v1, DuckDB, fastembed, pytest

**Key migrations:**
- `langgraph.prebuilt.create_react_agent` → `langchain.agents.create_agent`
- `ChatAnthropic`/`ChatOpenAI`/`ChatOllama` → `langchain.chat_models.init_chat_model`
- `prompt=` → `system_prompt=`
- Model format: `"claude-sonnet-4-20250514"` or `"anthropic:claude-sonnet-4-20250514"`

---

### Task 1: Replace _create_llm_client with init_chat_model

**Files:**
- Modify: `src/mao/agents.py`

**What changes:**
- Delete `_create_llm_client()` function (lines 86-140)
- Remove direct imports of `ChatAnthropic`, `ChatOpenAI`, `ChatOllama`
- Add `from langchain.chat_models import init_chat_model`
- All call sites use `init_chat_model(model_name, temperature=..., ...)` instead
- Provider is derived from model string or passed as `model_provider` kwarg
- OLLAMA_HOST support: pass `base_url` kwarg when provider is ollama

**New model initialization pattern:**
```python
from langchain.chat_models import init_chat_model

# Provider inferred from model string format "provider:model"
# or explicit: init_chat_model("gemma3:4b", model_provider="ollama", base_url=...)
def _create_model(provider: str, model_name: str, temperature: float = 0.0, stream: bool = False):
    kwargs = {"temperature": temperature, "streaming": stream}
    if provider == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            kwargs["base_url"] = ollama_host
    return init_chat_model(model_name, model_provider=provider, **kwargs)
```

**Step 1:** Write test that creates agent with new API (baseline, should pass before and after)
**Step 2:** Replace `_create_llm_client` with `_create_model` using `init_chat_model`
**Step 3:** Remove `ChatAnthropic`, `ChatOpenAI`, `ChatOllama` imports
**Step 4:** Update `Supervisor.__init_supervisor__()` to use `_create_model`
**Step 5:** Run tests: `uv run pytest tests/test_agents.py -v`
**Step 6:** Commit

---

### Task 2: Replace create_react_agent with langchain.agents.create_agent

**Files:**
- Modify: `src/mao/agents.py`

**What changes:**
- Remove `from langgraph.prebuilt import create_react_agent`
- Add `from langchain.agents import create_agent as lc_create_agent`
- In `Agent.init_agent()`: replace `create_react_agent(self.llm, tools, prompt=...)` with `lc_create_agent(self.llm, tools, system_prompt=...)`
- This fixes the deprecation warnings in tests

**Step 1:** Replace import and usage in `init_agent()`
**Step 2:** Run tests: `uv run pytest tests/test_agents.py -v`
**Step 3:** Verify no deprecation warnings
**Step 4:** Commit

---

### Task 3: Simplify Agent.__init__ — trees internal

**Files:**
- Modify: `src/mao/agents.py` (Agent class)

**What changes:**
- Remove `knowledge_tree` and `experience_tree` parameters from `Agent.__init__`
- Remove `callbacks`, `token_callback`, `max_tokens_trimmed`, `use_react_agent` parameters
- Keep them as internal attributes with sensible defaults
- `init_agent()` creates trees from `agent_name`:
  - `knowledge_{safe_name}` collection
  - `experience_{safe_name}` collection

```python
class Agent:
    def __init__(
        self,
        llm_instance: BaseChatModel,
        agent_name: str,
        tools: MCPClient | list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        stream: bool = False,
    ):
        self.llm = llm_instance
        self.name = agent_name
        self.configured_tools = tools
        self.loaded_tools: list[dict[str, Any]] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.knowledge_tree: KnowledgeTree | None = None
        self.experience_tree: ExperienceTree | None = None
        self.memory = MemorySaver()
        self.callbacks = DEFAULT_CALLBACKS
        self.stream = stream
        self.agent_runnable = None
```

In `init_agent()`:
```python
safe_name = self.name.replace("-", "_").replace(" ", "_")
self.knowledge_tree = await KnowledgeTree.create(
    collection_name=f"knowledge_{safe_name}"
)
self.experience_tree = await ExperienceTree.create(
    collection_name=f"experience_{safe_name}"
)
```

**Step 1:** Modify Agent class
**Step 2:** Run tests (some will fail)
**Step 3:** Commit

---

### Task 4: Simplify create_agent() factory

**Files:**
- Modify: `src/mao/agents.py` (create_agent function)

**New signature:**
```python
async def create_agent(
    provider: str,
    model_name: str,
    agent_name: str | None = None,
    system_prompt: str | None = None,
    tools: MCPClient | list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    stream: bool = False,
) -> Any:
    if not agent_name:
        sanitized = model_name.replace(".", "_").replace("/", "_")
        agent_name = f"{provider}_{sanitized}_agent"

    llm_instance = _create_model(provider, model_name, temperature, stream)

    agent_instance = Agent(
        llm_instance=llm_instance,
        agent_name=agent_name,
        tools=tools,
        system_prompt=system_prompt,
        stream=stream,
    )

    return await agent_instance.init_agent()
```

**Step 1:** Simplify function
**Step 2:** Run tests (some will fail due to removed params)
**Step 3:** Commit

---

### Task 5: Update tests to new API

**Files:**
- Modify: `tests/test_agents.py`
- Modify: `tests/conftest.py` (keep tree fixtures for storage tests only)

**What changes:**
- Remove `knowledge_tree`, `experience_tree` params from all agent test calls
- Remove `use_react_agent` param usage
- `test_agent_learning_retrieval`: simplify — test coherent multi-turn behavior instead of direct tree access
- `test_agent_with_mcp_tools`: remove `use_react_agent=True` (now default behavior)
- Keep `knowledge_tree`/`experience_tree` fixtures in conftest (still used by `test_storage.py`)

**Step 1:** Update all agent tests
**Step 2:** Run all tests: `uv run pytest -v`
**Step 3:** All pass
**Step 4:** Commit

---

### Task 6: Update API helpers and __init__.py

**Files:**
- Modify: `src/mao/api/helpers.py`
- Modify: `src/mao/__init__.py`

**helpers.py** — remove `use_react_agent`, `max_tokens_trimmed`, `llm_specific_kwargs` from `create_and_start_agent`:
```python
agent_app = await create_agent(
    provider=agent_config["provider"],
    model_name=agent_config["model_name"],
    agent_name=agent_config["name"],
    system_prompt=agent_config.get("system_prompt"),
    tools=tools,
)
```

**__init__.py** — remove storage exports:
```python
from .agents import create_agent, Agent, Supervisor
from .mcp import MCPClient, ToolConfig, ServerConfig

__all__ = [
    "create_agent",
    "Agent",
    "Supervisor",
    "MCPClient",
    "ToolConfig",
    "ServerConfig",
    "__version__",
]
```

**Step 1:** Update both files
**Step 2:** Run all tests: `uv run pytest -v`
**Step 3:** Commit

---

### Task 7: Update README and final verification

**Files:**
- Modify: `README.md`

**Quick Start becomes:**
```python
from mao import create_agent

agent = await create_agent(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    agent_name="assistant",
    system_prompt="You are a helpful data analyst.",
)

response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Analyze the latest data"}]}
)
```

**Step 1:** Update README
**Step 2:** Run full test suite: `uv run pytest -v`
**Step 3:** Run linting: `uv run ruff check src/ tests/`
**Step 4:** Commit
**Step 5:** Create PR
