"""Agent classes with Agent, Supervisor, and create_agent factory."""

import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv

from langchain.agents import create_agent as lc_create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_supervisor import create_supervisor

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mao.mcp import MCPClient
from mao.storage import ExperienceTree, KnowledgeTree

load_dotenv()

logger = logging.getLogger(__name__)


def _create_model(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    stream: bool = False,
) -> BaseChatModel:
    kwargs: dict[str, Any] = {"temperature": temperature, "streaming": stream}
    if provider.lower() == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            kwargs["base_url"] = ollama_host
    return init_chat_model(model_name, model_provider=provider, **kwargs)


async def load_mcp_tools(
    mcp_client: MCPClient | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if mcp_client is None:
        return []

    if isinstance(mcp_client, MCPClient):
        logger.debug("Loading tools from MCPClient...")
        try:
            tools = await mcp_client.get_tools()
            logger.debug("Loaded %d tools from MCP servers", len(tools))
            return tools
        except Exception as e:
            logger.error("Error loading MCP tools: %s", e)
            return []
    elif isinstance(mcp_client, list):
        return mcp_client
    else:
        logger.warning("Unexpected mcp_client type: %s", type(mcp_client))
        return []


async def _process_llm_response(response: Any) -> tuple[BaseMessage, str]:
    if isinstance(response, AIMessage):
        return response, str(response.content)
    content = str(response)
    return AIMessage(content=content), content


def _ensure_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return "\n".join(str(v) for v in val)
    return str(val)


def _dict_to_tool(d: dict[str, Any]) -> BaseTool:
    from pydantic import Field, create_model

    params = d.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    fields = {}
    for prop_name, prop_schema in properties.items():
        field_type = str
        prop_type = prop_schema.get("type", "string")
        if prop_type == "integer":
            field_type = int
        elif prop_type == "number":
            field_type = float
        elif prop_type == "boolean":
            field_type = bool
        elif prop_type == "array":
            field_type = list

        default = ... if prop_name in required else None
        desc = prop_schema.get("description", "")
        fields[prop_name] = (field_type, Field(default=default, description=desc))

    if not fields:
        fields["input"] = (str, Field(description="Input for the tool"))

    input_model = create_model(f"{d['name']}_Input", **fields)

    return StructuredTool.from_function(
        func=lambda **kwargs: f"Tool '{d['name']}' called with: {kwargs}",
        name=d["name"],
        description=d.get("description", ""),
        args_schema=input_model,
    )


def _dicts_to_tools(tools: list[Any]) -> list[Any]:
    result = []
    for t in tools:
        if callable(t) or isinstance(t, BaseTool):
            result.append(t)
        elif isinstance(t, dict) and "name" in t:
            try:
                result.append(_dict_to_tool(t))
            except Exception as e:
                logger.warning("Failed to convert dict tool '%s': %s", t.get("name"), e)
    return result


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
        self.stream = stream
        self.agent_runnable = None

    async def _load_mcp_tools(self) -> list[dict[str, Any]]:
        return await load_mcp_tools(self.configured_tools)

    async def _retrieve_context(self, query: str, k: int = 3) -> str:
        context_parts = []
        if self.knowledge_tree:
            knowledge_hits = await self.knowledge_tree.search_async(query, k=k)
            if knowledge_hits:
                context_parts.append(
                    "\nRelevant Knowledge:\n"
                    + "\n".join([h["page_content"] for h in knowledge_hits])
                )

        if self.experience_tree:
            experience_hits = await self.experience_tree.search_async(query, k=k)
            if experience_hits:
                context_parts.append(
                    "\nRelevant Experience:\n"
                    + "\n".join([h["page_content"] for h in experience_hits])
                )

        return "".join(context_parts).strip()

    async def _learn_experience(
        self, user_input: str, model_output: Any, tags: list[str] | None = None
    ) -> None:
        if not self.experience_tree:
            return

        model_output_str = _ensure_str(model_output)

        knowledge_id = None
        if self.knowledge_tree and user_input:
            knowledge_hits = await self.knowledge_tree.search_async(user_input, k=1)
            if knowledge_hits:
                knowledge_id = knowledge_hits[0].get("id")

        exp_text = f"User: {user_input}\nAgent: {model_output_str}"

        await self.experience_tree.learn_from_experience_async(
            exp_text, related_knowledge_id=knowledge_id, tags=tags
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_model_node(
        self, state: MessagesState, config: RunnableConfig | None = None
    ) -> dict[str, list[BaseMessage]]:
        user_input_raw = state["messages"][-1].content if state["messages"] else ""
        user_input = _ensure_str(user_input_raw)
        context_str = _ensure_str(await self._retrieve_context(user_input))

        system_content = self.system_prompt
        if context_str:
            system_content = f"{system_content}\n{context_str}"

        try:
            trimmed_messages_list = trim_messages(
                state["messages"],
                max_tokens=3000,
                strategy="last",
                token_counter=self.llm,
                include_system=True,
                start_on="human",
            )
        except Exception:
            logger.warning("Message trimming failed, using last 20 messages")
            trimmed_messages_list = list(state["messages"][-20:])
        messages_for_llm: list[BaseMessage] = [
            SystemMessage(content=system_content)
        ] + trimmed_messages_list

        try:
            llm = self.llm
            tools = _dicts_to_tools(self.loaded_tools)
            if tools and hasattr(llm, "bind_tools"):
                llm = llm.bind_tools(tools)

            invoked_response = await llm.ainvoke(messages_for_llm, config=config)
            response_message, learn_content = await _process_llm_response(
                invoked_response,
            )
            await self._learn_experience(user_input, learn_content)
            return {"messages": [response_message]}
        except Exception as e:
            logger.error("Agent '%s' model call failed: %s", self.name, e, exc_info=True)
            raise

    def _build_rag_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        if self.knowledge_tree:
            kt = self.knowledge_tree

            @tool
            async def retrieve_knowledge(query: str) -> str:
                """Search the agent's knowledge base for relevant information.

                Args:
                    query: Search query to find relevant knowledge
                """
                hits = await kt.search_async(query, k=3)
                if not hits:
                    return "No relevant knowledge found."
                return "\n".join(h["page_content"] for h in hits)

            tools.append(retrieve_knowledge)

        if self.experience_tree:
            et = self.experience_tree
            kt = self.knowledge_tree

            @tool
            async def retrieve_experience(query: str) -> str:
                """Search the agent's past experiences for relevant context.

                Args:
                    query: Search query to find relevant past experiences
                """
                hits = await et.search_async(query, k=3)
                if not hits:
                    return "No relevant experience found."
                return "\n".join(h["page_content"] for h in hits)

            tools.append(retrieve_experience)

            @tool
            async def save_experience(summary: str) -> str:
                """Save a summary of what was learned or accomplished for future reference.

                Args:
                    summary: Brief summary of the interaction outcome or lesson learned
                """
                knowledge_id = None
                if kt:
                    hits = await kt.search_async(summary, k=1)
                    if hits:
                        knowledge_id = hits[0].get("id")
                await et.learn_from_experience_async(
                    summary, related_knowledge_id=knowledge_id
                )
                return "Experience saved."

            tools.append(save_experience)

        return tools

    async def init_agent(self):
        try:
            safe_name = self.name.replace("-", "_").replace(" ", "_")
            self.knowledge_tree = await KnowledgeTree.create(
                collection_name=f"knowledge_{safe_name}"
            )
            self.experience_tree = await ExperienceTree.create(
                collection_name=f"experience_{safe_name}"
            )

            self.loaded_tools = await self._load_mcp_tools()
            rag_tools = self._build_rag_tools()
            self.loaded_tools.extend(rag_tools)
            logger.info("Agent '%s' loaded %d tools (%d RAG)", self.name, len(self.loaded_tools), len(rag_tools))

            if self.loaded_tools:
                all_tools = _dicts_to_tools(self.loaded_tools)
                self.agent_runnable = lc_create_agent(
                    model=self.llm,
                    tools=all_tools,
                    system_prompt=self.system_prompt,
                    checkpointer=self.memory,
                    name=self.name,
                )
                logger.info(
                    "Agent '%s' created with create_agent and %d tools.",
                    self.name,
                    len(self.loaded_tools),
                )
                return self.agent_runnable

            workflow = StateGraph(MessagesState)
            workflow.add_node("model_node", self._call_model_node)
            workflow.add_edge(START, "model_node")
            workflow.add_edge("model_node", END)

            self.agent_runnable = workflow.compile(
                checkpointer=self.memory, name=self.name
            )
            logger.info("Agent '%s' compiled as custom graph (no tools).", self.name)
            return self.agent_runnable
        except Exception as e:
            logger.error("Failed to initialize agent '%s': %s", self.name, e, exc_info=True)
            raise

    def get_compiled_app(self):
        if not self.agent_runnable:
            raise RuntimeError("Agent not initialized. Call init_agent() first.")
        return self.agent_runnable


class Supervisor:
    def __init__(
        self,
        agents: list[Any],
        supervisor_provider: str,
        supervisor_model_name: str,
        supervisor_system_prompt: str,
        supervisor_tools: MCPClient | list[dict[str, Any]] | None = None,
        add_handoff_back_messages: bool = True,
        parallel_tool_calls: bool = True,
        **supervisor_kwargs: Any,
    ):
        self.agents = agents
        self.supervisor_provider = supervisor_provider
        self.supervisor_model_name = supervisor_model_name
        self.llm: BaseChatModel | None = None
        self.prompt = supervisor_system_prompt
        self.supervisor_tools = supervisor_tools
        self.supervisor_kwargs = {
            "add_handoff_back_messages": add_handoff_back_messages,
            "parallel_tool_calls": parallel_tool_calls,
            **supervisor_kwargs,
        }
        self.memory = MemorySaver()
        self.app = None

    async def _load_supervisor_mcp_tools(self) -> list[dict[str, Any]]:
        return await load_mcp_tools(self.supervisor_tools)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _create_supervisor_workflow(self) -> None:
        try:
            tools = _dicts_to_tools(await self._load_supervisor_mcp_tools())
            logger.info("Supervisor loaded %d tools", len(tools))

            workflow = create_supervisor(
                self.agents,
                model=self.llm,
                prompt=self.prompt,
                tools=tools if tools else None,
                **self.supervisor_kwargs,
            )
            self.app = workflow.compile(
                checkpointer=self.memory, name="global_supervisor"
            )
        except Exception as e:
            logger.error("Supervisor workflow initialization failed: %s", e, exc_info=True)
            raise

    async def init_supervisor(self):
        self.llm = _create_model(
            provider=self.supervisor_provider,
            model_name=self.supervisor_model_name,
            temperature=0.0,
        )
        await self._create_supervisor_workflow()
        logger.info("Supervisor initialized and compiled successfully.")
        return self.app

    async def invoke(self, messages: list[dict], thread_id: str | None = None) -> Any:
        if self.app is None:
            raise RuntimeError(
                "Supervisor must be initialized. Call init_supervisor() first."
            )
        config_dict = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
        return await self.app.ainvoke({"messages": messages}, config=config_dict)


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
        sanitized_model_name = model_name.replace(".", "_").replace("/", "_")
        agent_name = f"{provider}_{sanitized_model_name}_agent"

    llm_instance = _create_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        stream=stream,
    )

    agent_instance = Agent(
        llm_instance=llm_instance,
        agent_name=agent_name,
        tools=tools,
        system_prompt=system_prompt,
        stream=stream,
    )

    compiled_app = await agent_instance.init_agent()
    return compiled_app
