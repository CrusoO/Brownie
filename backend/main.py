import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

import chromadb
from chromadb.api.types import Documents, Embeddings
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_tools import execute_play_music, execute_web_search
from chat_history import ChatHistoryStore, ChatTurn

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - the fallback brain keeps the API runnable.
    AIMessage = None
    HumanMessage = None
    SystemMessage = None
    ChatOpenAI = None

from langgraph.graph import END, START, StateGraph

from voice_pipeline import (
    VoiceState,
    get_pipeline,
    cleanup_pipeline,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TraceSink = Callable[[dict[str, Any]], Awaitable[None]]
trace_sink: ContextVar[Optional[TraceSink]] = ContextVar("trace_sink", default=None)


class Settings(BaseModel):
    app_name: str = "Brownie"
    memory_dir: str = Field(default_factory=lambda: os.getenv("BROWNIE_MEMORY_DIR", "./data/chroma"))
    memory_collection: str = Field(default_factory=lambda: os.getenv("BROWNIE_COLLECTION", "brownie_memory"))
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "BROWNIE_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000",
            ).split(",")
            if origin.strip()
        ]
    )
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    openai_model: str = Field(default_factory=lambda: os.getenv("BROWNIE_MODEL", "gpt-4o-mini"))

    @property
    def llm_api_key(self) -> Optional[str]:
        return self.groq_api_key or self.openai_api_key

    @property
    def llm_base_url(self) -> Optional[str]:
        if self.openai_base_url:
            return self.openai_base_url
        if self.groq_api_key:
            return "https://api.groq.com/openai/v1"
        return None

    @property
    def llm_model(self) -> str:
        if os.getenv("BROWNIE_MODEL"):
            return self.openai_model
        if self.groq_api_key:
            return "llama-3.3-70b-versatile"
        return self.openai_model
    sandbox_image: str = Field(default_factory=lambda: os.getenv("BROWNIE_SANDBOX_IMAGE", "python:3.12-alpine"))
    sandbox_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("BROWNIE_SANDBOX_TIMEOUT_SECONDS", "8"))
    )
    max_python_chars: int = Field(default_factory=lambda: int(os.getenv("BROWNIE_MAX_PYTHON_CHARS", "12000")))
    chat_history_max_turns: int = Field(
        default_factory=lambda: int(os.getenv("BROWNIE_CHAT_HISTORY_TURNS", "40"))
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=24000)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    response: str
    trace: List[Dict[str, Any]]
    route: str
    memory_ids: List[str] = Field(default_factory=list)


class WorkflowCreate(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=120)
    trigger: str = Field(min_length=1, max_length=180)
    steps: List[str] = Field(min_length=1, max_length=24)


class Workflow(BaseModel):
    id: str
    session_id: str
    name: str
    trigger: str
    steps: List[str]
    created_at: str
    updated_at: str
    run_count: int = 0


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    memory_dir: str
    llm_enabled: bool
    sandbox_image: str


class VoiceStateResponse(BaseModel):
    """Voice pipeline state."""
    state: str
    microphone_enabled: bool
    camera_enabled: bool
    active_clone: Optional[str]
    voices_available: List[str]
    timestamp: str


class VoiceControlRequest(BaseModel):
    """Voice control request."""
    action: Literal["enable", "disable"]
    sensor: Literal["microphone", "camera"]


class VoiceCloneLoadRequest(BaseModel):
    """Load voice clone profile."""
    name: str
    samples: List[str]  # Paths or URLs to audio samples


class VoiceCloneResponse(BaseModel):
    """Voice clone loaded response."""
    name: str
    speaker_id: str
    samples_count: int
    created_at: str


class BrownieState(TypedDict, total=False):
    run_id: str
    session_id: str
    user_message: str
    chat_history: List[ChatTurn]
    memory_context: List[str]
    workflow_context: List[str]
    route: Literal["talk", "tool"]
    tool_name: Optional[str]
    tool_input: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]
    response: str
    trace: List[Dict[str, Any]]
    memory_ids: List[str]
    llm_error: Optional[str]


class HashEmbeddingFunction:
    """Small local embedding function so Chroma works without external downloads."""

    dimensions = 384

    def name(self) -> str:
        return "brownie_hash_embedding"

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(document or "") for document in input]

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_']+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class VectorMemory:
    def __init__(self, settings: Settings) -> None:
        os.makedirs(settings.memory_dir, exist_ok=True)
        self.collection = chromadb.PersistentClient(path=settings.memory_dir).get_or_create_collection(
            name=settings.memory_collection,
            embedding_function=HashEmbeddingFunction(),
            metadata={"description": "Persistent long-term action/outcome memory for Brownie."},
        )

    def search(self, query: str, session_id: str, limit: int = 5) -> List[str]:
        if not query.strip():
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"session_id": session_id},
            )
        except Exception:
            return []

        documents = results.get("documents") or [[]]
        return [document for document in documents[0] if document]

    def remember(
        self,
        *,
        session_id: str,
        action: str,
        outcome: str,
        route: str,
        tool_name: Optional[str],
    ) -> str:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        document = f"Action: {action}\nOutcome: {outcome}\nRoute: {route}\nTool: {tool_name or 'none'}"
        self.collection.add(
            ids=[memory_id],
            documents=[document],
            metadatas=[
                {
                    "session_id": session_id,
                    "route": route,
                    "tool_name": tool_name or "",
                    "created_at": timestamp,
                }
            ],
        )
        return memory_id


class WorkflowStore:
    def __init__(self, settings: Settings) -> None:
        os.makedirs(settings.memory_dir, exist_ok=True)
        self.path = os.path.join(settings.memory_dir, "workflows.json")
        self.lock = threading.Lock()

    def list(self, session_id: str) -> List[Workflow]:
        with self.lock:
            workflows = self._read()
        return [workflow for workflow in workflows if workflow.session_id == session_id]

    def create(self, payload: WorkflowCreate) -> Workflow:
        timestamp = datetime.now(timezone.utc).isoformat()
        workflow = Workflow(
            id=str(uuid.uuid4()),
            session_id=payload.session_id,
            name=payload.name.strip(),
            trigger=payload.trigger.strip(),
            steps=[step.strip() for step in payload.steps if step.strip()],
            created_at=timestamp,
            updated_at=timestamp,
        )
        if not workflow.steps:
            raise ValueError("At least one non-empty step is required.")

        with self.lock:
            workflows = self._read()
            workflows.append(workflow)
            self._write(workflows)
        return workflow

    def delete(self, workflow_id: str, session_id: str) -> None:
        with self.lock:
            workflows = self._read()
            next_workflows = [
                workflow
                for workflow in workflows
                if not (workflow.id == workflow_id and workflow.session_id == session_id)
            ]
            if len(next_workflows) == len(workflows):
                raise KeyError(workflow_id)
            self._write(next_workflows)

    def mark_run(self, workflow_id: str, session_id: str) -> Workflow:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.lock:
            workflows = self._read()
            for index, workflow in enumerate(workflows):
                if workflow.id == workflow_id and workflow.session_id == session_id:
                    updated = workflow.model_copy(
                        update={
                            "run_count": workflow.run_count + 1,
                            "updated_at": timestamp,
                        }
                    )
                    workflows[index] = updated
                    self._write(workflows)
                    return updated
        raise KeyError(workflow_id)

    def find_matches(self, message: str, session_id: str, limit: int = 3) -> List[Workflow]:
        normalized = self._normalize(message)
        if not normalized:
            return []

        scored: List[Tuple[int, Workflow]] = []
        for workflow in self.list(session_id):
            trigger = self._normalize(workflow.trigger)
            name = self._normalize(workflow.name)
            score = 0
            if normalized == trigger or normalized == name:
                score = 100
            elif trigger and trigger in normalized:
                score = 80
            elif name and name in normalized:
                score = 60

            if score:
                scored.append((score, workflow))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [workflow for _, workflow in scored[:limit]]

    def format_for_prompt(self, workflow: Workflow) -> str:
        steps = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(workflow.steps))
        return (
            f"Taught workflow: {workflow.name}\n"
            f"Trigger: {workflow.trigger}\n"
            f"Run count: {workflow.run_count}\n"
            f"Steps:\n{steps}"
        )

    def _read(self) -> List[Workflow]:
        if not os.path.exists(self.path):
            return []

        try:
            with open(self.path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return []

        workflows = payload.get("workflows", []) if isinstance(payload, dict) else []
        return [Workflow(**workflow) for workflow in workflows if isinstance(workflow, dict)]

    def _write(self, workflows: List[Workflow]) -> None:
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(
                {"workflows": [workflow.model_dump() for workflow in workflows]},
                file,
                indent=2,
            )
        os.replace(temp_path, self.path)

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())


class SimpleRequestCache:
    """Simple LRU cache for recent requests to reduce duplicate processing."""
    
    def __init__(self, max_size: int = 50, ttl_seconds: int = 300) -> None:
        self.cache: Dict[str, Tuple[ChatResponse, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def get_key(self, request: ChatRequest) -> str:
        """Generate cache key from request."""
        content = f"{request.session_id}:{request.message}".lower().strip()
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, request: ChatRequest) -> Optional[ChatResponse]:
        """Get cached response if exists and not expired."""
        with self.lock:
            key = self.get_key(request)
            if key not in self.cache:
                return None
            
            response, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                return None
            
            return response
    
    def set(self, request: ChatRequest, response: ChatResponse) -> None:
        """Cache response with TTL."""
        with self.lock:
            # Keep cache size manageable
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            key = self.get_key(request)
            self.cache[key] = (response, time.time())


class DockerSandbox:
    def __init__(self, settings: Settings) -> None:
        self.image = settings.sandbox_image
        self.timeout_seconds = settings.sandbox_timeout_seconds
        self.max_python_chars = settings.max_python_chars

    async def run_python(self, code: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_python_blocking, code)

    def _run_python_blocking(self, code: str) -> dict[str, Any]:
        if len(code) > self.max_python_chars:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Python payload exceeds {self.max_python_chars} characters.",
                "exit_code": 1,
            }

        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "128m",
            "--pids-limit",
            "64",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--workdir",
            "/tmp",
            self.image,
            "python",
            "-I",
            "-S",
            "-",
        ]

        try:
            completed = subprocess.run(
                command,
                input=code,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "Docker CLI is not available. Install Docker or run the backend container with Docker access.",
                "exit_code": 127,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "stdout": exc.stdout or "",
                "stderr": f"Sandbox timed out after {self.timeout_seconds} seconds.",
                "exit_code": 124,
            }

        return {
            "ok": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }


class BrownieAgent:
    def __init__(
        self,
        settings: Settings,
        memory: VectorMemory,
        workflows: WorkflowStore,
        sandbox: DockerSandbox,
        chat_history: ChatHistoryStore,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.workflows = workflows
        self.sandbox = sandbox
        self.chat_history = chat_history
        self.llm = self._build_llm()
        self.graph = self._build_graph()

    def _build_llm(self) -> Any:
        api_key = self.settings.llm_api_key
        if not api_key and self.settings.llm_base_url:
            api_key = "ollama"

        if not api_key or ChatOpenAI is None:
            return None

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "api_key": api_key,
            "temperature": 0.2,
            "streaming": True,
        }
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url

        return ChatOpenAI(**kwargs)

    def _router_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python_sandbox",
                    "description": "Run Python code in an isolated Docker sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python source code to execute"},
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "play_music",
                    "description": "Play a song or music. Use when the user wants to hear music.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Song name, artist, or description",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def _build_graph(self) -> Any:
        graph = StateGraph(BrownieState)
        graph.add_node("load_memory", self.load_memory)
        graph.add_node("decide", self.decide)
        graph.add_node("run_tool", self.run_tool)
        graph.add_node("talk", self.talk)
        graph.add_node("persist_memory", self.persist_memory)

        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "decide")
        graph.add_conditional_edges(
            "decide",
            lambda state: state.get("route", "talk"),
            {"tool": "run_tool", "talk": "talk"},
        )
        graph.add_edge("run_tool", "talk")
        graph.add_edge("talk", "persist_memory")
        graph.add_edge("persist_memory", END)
        return graph.compile()

    async def run(self, request: ChatRequest, emit: Optional[TraceSink] = None) -> ChatResponse:
        run_id = str(uuid.uuid4())
        initial_state: BrownieState = {
            "run_id": run_id,
            "session_id": request.session_id,
            "user_message": request.message,
            "trace": [],
            "memory_ids": [],
        }

        token = trace_sink.set(emit)
        try:
            final_state = await self.graph.ainvoke(initial_state)
        finally:
            trace_sink.reset(token)

        return ChatResponse(
            run_id=run_id,
            session_id=request.session_id,
            response=final_state.get("response", ""),
            trace=final_state.get("trace", []),
            route=final_state.get("route", "talk"),
            memory_ids=final_state.get("memory_ids", []),
        )

    async def load_memory(self, state: BrownieState) -> BrownieState:
        memories = await asyncio.to_thread(
            self.memory.search,
            state["user_message"],
            state["session_id"],
            3,
        )
        workflow_matches = await asyncio.to_thread(
            self.workflows.find_matches,
            state["user_message"],
            state["session_id"],
            2,
        )
        workflow_context = [self.workflows.format_for_prompt(workflow) for workflow in workflow_matches]
        history = await asyncio.to_thread(
            self.chat_history.list,
            state["session_id"],
            self.settings.chat_history_max_turns,
        )
        event = await self._event(
            state,
            "memory.loaded",
            "Retrieving context from long-term memory.",
            {
                "matches": len(memories),
                "workflows": len(workflow_context),
                "chat_turns": len(history),
            },
        )
        return {
            "chat_history": history,
            "memory_context": memories,
            "workflow_context": workflow_context,
            "trace": [*state.get("trace", []), event],
        }

    async def decide(self, state: BrownieState) -> BrownieState:
        llm_error = None
        message = state["user_message"]
        heuristic_decision = self._fallback_decision(message)

        if heuristic_decision.get("route") == "tool":
            decision = heuristic_decision
        elif self.llm:
            try:
                decision = await self._llm_decision(state)
            except Exception as exc:
                llm_error = self._format_llm_error(exc)
                decision = heuristic_decision
        else:
            decision = heuristic_decision

        event_data = {
            "route": decision["route"],
            "tool_name": decision.get("tool_name"),
        }
        if llm_error:
            event_data["llm_error"] = llm_error

        event = await self._event(
            state,
            "agent.route",
            "Planning the next action.",
            event_data,
        )
        updates: BrownieState = {
            "route": decision["route"],
            "tool_name": decision.get("tool_name"),
            "tool_input": decision.get("tool_input", {}),
            "trace": [*state.get("trace", []), event],
        }
        if llm_error:
            updates["llm_error"] = llm_error
        return updates

    async def run_tool(self, state: BrownieState) -> BrownieState:
        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input", {})
        tool_labels = {
            "python_sandbox": "Running Python inside the Docker sandbox.",
            "web_search": "Searching the web for current information.",
            "play_music": "Finding music to play.",
        }
        start_event = await self._event(
            state,
            "tool.started",
            tool_labels.get(tool_name or "", "Running tool."),
            {"tool_name": tool_name},
        )
        state = {**state, "trace": [*state.get("trace", []), start_event]}

        if tool_name == "python_sandbox":
            result = await self.sandbox.run_python(str(tool_input.get("code", "")))
        elif tool_name == "web_search":
            result = await asyncio.to_thread(
                execute_web_search,
                str(tool_input.get("query", state["user_message"])),
            )
        elif tool_name == "play_music":
            result = await asyncio.to_thread(
                execute_play_music,
                str(tool_input.get("query", state["user_message"])),
            )
        else:
            result = {
                "ok": False,
                "stdout": "",
                "stderr": f"Unknown tool: {tool_name}",
                "exit_code": 1,
            }

        finish_data: dict[str, Any] = {
            "tool_name": tool_name,
            "ok": result.get("ok", False),
        }
        if "exit_code" in result:
            finish_data["exit_code"] = result["exit_code"]

        finish_event = await self._event(
            state,
            "tool.finished",
            "Tool execution finished.",
            finish_data,
        )
        return {
            "tool_result": result,
            "trace": [*state.get("trace", []), finish_event],
        }

    async def talk(self, state: BrownieState) -> BrownieState:
        llm_error = state.get("llm_error")
        if self.llm and not llm_error:
            try:
                response = await self._llm_response(state, stream_tokens=True)
            except Exception as exc:
                llm_error = self._format_llm_error(exc)
                response = self._fallback_response(state, llm_error=llm_error)
        else:
            response = self._fallback_response(state, llm_error=llm_error)

        event_data: dict[str, Any] = {"chars": len(response)}
        if llm_error:
            event_data["llm_error"] = llm_error

        event = await self._event(
            state,
            "agent.response",
            "Prepared the final response.",
            event_data,
        )
        updates: BrownieState = {
            "response": response,
            "trace": [*state.get("trace", []), event],
        }
        if llm_error:
            updates["llm_error"] = llm_error
        return updates

    async def persist_memory(self, state: BrownieState) -> BrownieState:
        memory_id = await asyncio.to_thread(
            self.memory.remember,
            session_id=state["session_id"],
            action=state["user_message"],
            outcome=state.get("response", ""),
            route=state.get("route", "talk"),
            tool_name=state.get("tool_name"),
        )
        await asyncio.to_thread(
            self.chat_history.append_exchange,
            state["session_id"],
            state["user_message"],
            state.get("response", ""),
        )
        event = await self._event(
            state,
            "memory.saved",
            "Saving this turn to long-term memory.",
            {"memory_id": memory_id},
        )
        return {
            "memory_ids": [*state.get("memory_ids", []), memory_id],
            "trace": [*state.get("trace", []), event],
        }

    def _session_context_block(self, state: BrownieState) -> str:
        memory = "\n\n".join(state.get("memory_context", [])) or "No relevant long-term memory."
        workflows = "\n\n".join(state.get("workflow_context", [])) or "No matching taught workflows."
        tool_result = state.get("tool_result")
        tool_summary = json.dumps(tool_result, ensure_ascii=True) if tool_result else "No tool was used."
        return (
            f"Relevant long-term memory:\n{memory}\n\n"
            f"Taught workflows:\n{workflows}\n\n"
            f"Tool result this turn:\n{tool_summary}"
        )

    def _build_llm_messages(self, state: BrownieState, system: str) -> List[Any]:
        if SystemMessage is None or HumanMessage is None:
            return []

        messages: List[Any] = [
            SystemMessage(content=f"{system}\n\n{self._session_context_block(state)}"),
        ]
        if AIMessage is not None:
            messages.extend(self.chat_history.to_langchain_messages(state.get("chat_history", [])))
        messages.append(HumanMessage(content=state["user_message"]))
        return messages

    async def _event(
        self,
        state: BrownieState,
        event_type: str,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        event = {
            "id": str(uuid.uuid4()),
            "run_id": state["run_id"],
            "type": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        sink = trace_sink.get()
        if sink:
            await sink(event)
        return event

    async def _emit_token(self, state: BrownieState, token: str) -> None:
        sink = trace_sink.get()
        if not sink or not token:
            return
        await sink(
            {
                "id": str(uuid.uuid4()),
                "run_id": state["run_id"],
                "type": "agent.response.token",
                "message": token,
                "data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _llm_decision(self, state: BrownieState) -> dict[str, Any]:
        if HumanMessage is None or SystemMessage is None:
            return self._fallback_decision(state["user_message"])

        system = (
            "You are Brownie's router. Call a tool only when needed. "
            "Use python_sandbox when the user wants Python executed or provides code. "
            "Use web_search for current events, facts, or anything that needs up-to-date web info. "
            "Use play_music when the user wants to hear a song or music. "
            "Use the conversation history when the user refers to earlier messages (e.g. 'what did I say about…'). "
            "If no tool is needed, respond with a short plain-text plan and do not call tools."
        )
        messages = self._build_llm_messages(state, system)

        llm_router = self.llm.bind_tools(self._router_tools())
        raw = await llm_router.ainvoke(messages)
        tool_calls = getattr(raw, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            tool_name = call.get("name")
            tool_input = call.get("args") if isinstance(call.get("args"), dict) else {}
            if tool_name in {"python_sandbox", "web_search", "play_music"}:
                return {
                    "route": "tool",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                }

        return {
            "route": "talk",
            "tool_name": None,
            "tool_input": {},
        }

    async def _llm_response(self, state: BrownieState, stream_tokens: bool = False) -> str:
        if HumanMessage is None or SystemMessage is None:
            return self._fallback_response(state)

        system = (
            "You are Brownie, a smart, friendly, and slightly witty AI assistant. "
            "Speak like a human friend, not a robot. Keep responses short and natural. "
            "Be warm, casual, and helpful. Add small emotions when appropriate (like 'hmm', 'okay got it'). "
            "If the user gives a command, respond briefly and confirm action. "
            "If it's a question, answer clearly but casually. Never be overly long unless asked. "
            "If user says 'stop', immediately stop speaking. "
            "Use the conversation history in this session to answer follow-ups (e.g. 'what did I ask before?', "
            "'remind me about…', 'the thing we discussed'). Quote or paraphrase earlier turns when helpful. "
            "Use long-term memory and tool results for facts; use chat history for what was said in this chat. "
            "When a taught workflow matches the user request, execute the workflow conversationally: "
            "state what you can do now, use any available tool result, and ask for confirmation only for unsafe or external actions. "
            "Do not claim you changed files, opened apps, sent messages, or contacted services unless a tool result proves it. "
            "Do not reveal hidden chain-of-thought; summarize actions and outcomes instead. "
            "Examples of good responses: 'Got it, turning on the camera 📸' or 'Alright, I'll stay quiet now.'"
        )
        messages = self._build_llm_messages(state, system)

        if not stream_tokens:
            raw = await self.llm.ainvoke(messages)
            return str(raw.content).strip()

        parts: list[str] = []
        async for chunk in self.llm.astream(messages):
            token = self._chunk_text(chunk)
            if not token:
                continue
            parts.append(token)
            await self._emit_token(state, token)
        return "".join(parts).strip()

    def _chunk_text(self, chunk: Any) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        return str(content or "")

    def _fallback_decision(self, message: str) -> dict[str, Any]:
        code = self._extract_python(message)
        if code:
            return {
                "route": "tool",
                "tool_name": "python_sandbox",
                "tool_input": {"code": code},
            }

        lowered = message.lower()
        if re.search(r"\b(play|start)\b.*\b(music|song|track)\b", lowered) or re.search(
            r"\bplay\b.+\b(on spotify|for me)\b", lowered
        ):
            query = re.sub(
                r"^(please\s+)?(play|start)\s+(the\s+)?(song|music|track)?\s*",
                "",
                message,
                flags=re.IGNORECASE,
            ).strip(" .")
            return {
                "route": "tool",
                "tool_name": "play_music",
                "tool_input": {"query": query or message},
            }

        if re.search(r"\b(search|look up|google|find out|what is|who is|latest)\b", lowered):
            return {
                "route": "tool",
                "tool_name": "web_search",
                "tool_input": {"query": message},
            }

        return {
            "route": "talk",
            "tool_name": None,
            "tool_input": {},
        }

    def _fallback_response(self, state: BrownieState, llm_error: Optional[str] = None) -> str:
        tool_result = state.get("tool_result")
        tool_name = state.get("tool_name")
        if tool_result and tool_name == "web_search":
            if tool_result.get("ok"):
                lines = [
                    f"- {item.get('title', 'Result')}: {item.get('snippet', '')} ({item.get('url', '')})"
                    for item in tool_result.get("results", [])[:3]
                ]
                return "Here's what I found:\n\n" + ("\n".join(lines) if lines else "No results.")
            return f"Search failed: {tool_result.get('error', 'unknown error')}"

        if tool_result and tool_name == "play_music" and tool_result.get("ok"):
            return (
                f"Here's music for \"{tool_result.get('query', 'your request')}\":\n"
                f"Spotify: {tool_result.get('spotify_url')}\n"
                f"YouTube: {tool_result.get('youtube_url')}"
            )

        if tool_result:
            stdout = (tool_result.get("stdout") or "").strip()
            stderr = (tool_result.get("stderr") or "").strip()
            if tool_result.get("ok"):
                return f"Nice! Code ran successfully. Here's what it output:\n\n{stdout}"
            return f"Hmm, that code had some issues. Here's what went wrong:\n\n{stderr or stdout}"

        if llm_error:
            return (
                f"Oops, ran into a little hiccup with the AI service: {llm_error}\n\n"
                "I'm still here and ready to help! Check your API setup if you want full AI responses."
            )

        if state.get("workflow_context"):
            return (
                f"Hey, I found a workflow that matches what you asked for!\n\n"
                f"{state['workflow_context'][0]}"
            )

        memory_hint = ""
        if state.get("memory_context"):
            memory_hint = "\n\nI remember some related stuff from before - I'll use that to help you better."

        return (
            f"Hey there! I'm running and ready to help. "
            f"Add GROQ_API_KEY (free at console.groq.com) for fast AI responses, "
            f"or I can still search the web, queue music links, run Python, and remember things for you. {memory_hint}"
        )

    def _format_llm_error(self, exc: Exception) -> str:
        text = str(exc)
        lower_text = text.lower()
        if "insufficient_quota" in lower_text or "exceeded your current quota" in lower_text:
            return "looks like we're out of API credits - add some billing to keep the conversation going!"
        if "rate limit" in lower_text or exc.__class__.__name__ == "RateLimitError":
            return "whoa, hitting the rate limit - let's wait a moment and try again"
        if "authentication" in lower_text or "api key" in lower_text:
            return "authentication hiccup - double-check your API key and base URL in the .env file"
        return f"ran into a {exc.__class__.__name__} error: {text[:240]}"

    def _extract_python(self, message: str) -> Optional[str]:
        fenced = re.search(r"```(?:python|py)?\s*(.*?)```", message, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        inline = re.search(
            r"(?:run|execute)\s+(?:this\s+)?(?:python|py)\s*:?\s*(.+)",
            message,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if inline:
            return inline.group(1).strip()

        return None

    def _json_from_text(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    def _normalize_decision(self, decision: dict[str, Any], message: str) -> dict[str, Any]:
        route = decision.get("route")
        tool_name = decision.get("tool_name")
        tool_input = decision.get("tool_input") if isinstance(decision.get("tool_input"), dict) else {}
        code = str(tool_input.get("code", "")).strip()

        if route == "tool" and tool_name == "python_sandbox" and code:
            return {
                "route": "tool",
                "tool_name": "python_sandbox",
                "tool_input": {"code": code},
            }

        return self._fallback_decision(message) if self._extract_python(message) else {
            "route": "talk",
            "tool_name": None,
            "tool_input": {},
        }


settings = Settings()
memory = VectorMemory(settings)
workflows = WorkflowStore(settings)
sandbox = DockerSandbox(settings)
chat_history = ChatHistoryStore(settings.memory_dir, max_turns=settings.chat_history_max_turns)
agent = BrownieAgent(settings, memory, workflows, sandbox, chat_history)
request_cache = SimpleRequestCache(max_size=50, ttl_seconds=300)


def _should_use_request_cache(request: ChatRequest) -> bool:
    """Skip cache when the session has prior turns so follow-ups stay contextual."""
    return not chat_history.has_history(request.session_id)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        memory_dir=settings.memory_dir,
        llm_enabled=agent.llm is not None,
        sandbox_image=settings.sandbox_image,
    )


@app.get("/chat/history", response_model=List[ChatTurn])
async def get_chat_history(session_id: str = "default") -> List[ChatTurn]:
    return chat_history.list(session_id)


@app.delete("/chat/history", status_code=204)
async def clear_chat_history(session_id: str = "default") -> None:
    chat_history.clear(session_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if _should_use_request_cache(request):
        cached = request_cache.get(request)
        if cached:
            return cached

    response = await agent.run(request)
    if _should_use_request_cache(request):
        request_cache.set(request, response)
    return response


@app.get("/stream/chat")
async def stream_chat(
    message: str = Query(..., min_length=1, max_length=24000),
    session_id: str = Query(default="default", min_length=1, max_length=128),
) -> StreamingResponse:
    request = ChatRequest(message=message, session_id=session_id)

    async def event_stream() -> Any:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def emit(event: dict[str, Any]) -> None:
            await queue.put(event)

        run_task = asyncio.create_task(agent.run(request, emit=emit))

        while not run_task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
            except asyncio.TimeoutError:
                continue

        response = await run_task
        if _should_use_request_cache(request):
            request_cache.set(request, response)
        yield f"data: {json.dumps({'type': 'final', 'response': response.model_dump()}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/workflows", response_model=List[Workflow])
async def list_workflows(session_id: str = "default") -> List[Workflow]:
    return workflows.list(session_id)


@app.post("/workflows", response_model=Workflow)
async def create_workflow(payload: WorkflowCreate) -> Workflow:
    try:
        return workflows.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workflows/{workflow_id}/run", response_model=Workflow)
async def mark_workflow_run(workflow_id: str, session_id: str = "default") -> Workflow:
    try:
        return workflows.mark_run(workflow_id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workflow not found.") from exc


@app.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, session_id: str = "default") -> None:
    try:
        workflows.delete(workflow_id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workflow not found.") from exc


# ===== VOICE CONTROL ENDPOINTS =====


@app.get("/voice/state", response_model=VoiceStateResponse)
async def get_voice_state() -> VoiceStateResponse:
    """Get current voice pipeline state."""
    pipeline = get_pipeline()
    state_dict = pipeline.get_state_dict()
    return VoiceStateResponse(**state_dict)


@app.post("/voice/control")
async def control_voice(request: VoiceControlRequest) -> dict[str, str]:
    """Control microphone or camera."""
    pipeline = get_pipeline()
    
    if request.sensor == "microphone":
        if request.action == "enable":
            await pipeline.enable_microphone()
            return {"status": "Alright, I'm listening now 🎤"}
        else:
            await pipeline.disable_microphone()
            return {"status": "Okay, I'll stay quiet."}
    
    elif request.sensor == "camera":
        if request.action == "enable":
            await pipeline.enable_camera()
            return {"status": "Alright, camera's on 📸"}
        else:
            await pipeline.disable_camera()
            return {"status": "Camera's off."}
    
    return {"status": "Unknown command"}


@app.post("/voice/clone", response_model=VoiceCloneResponse)
async def load_voice_clone(request: VoiceCloneLoadRequest) -> VoiceCloneResponse:
    """Load voice clone profile."""
    pipeline = get_pipeline()
    pipeline.load_voice_clone(request.name, request.samples)
    
    profile = pipeline.voice_profiles[request.name]
    return VoiceCloneResponse(
        name=profile.name,
        speaker_id=profile.speaker_id,
        samples_count=len(profile.samples),
        created_at=profile.created_at,
    )


@app.post("/voice/interrupt")
async def interrupt_speech() -> dict[str, str]:
    """Interrupt current speech immediately."""
    pipeline = get_pipeline()
    await pipeline.interrupt_speech()
    return {"status": "Speech interrupted"}


# ===== END VOICE CONTROL ENDPOINTS =====


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            request = ChatRequest(**payload)
            
            if _should_use_request_cache(request):
                cached = request_cache.get(request)
                if cached:
                    await websocket.send_json({"type": "final", "response": cached.model_dump()})
                    continue

            async def emit(event: dict[str, Any]) -> None:
                if event.get("type") == "agent.response.token":
                    await websocket.send_json({"type": "token", "token": event.get("message", "")})
                    return
                await websocket.send_json({"type": "trace", "event": event})

            response = await agent.run(request, emit=emit)
            if _should_use_request_cache(request):
                request_cache.set(request, response)
            await websocket.send_json({"type": "final", "response": response.model_dump()})
    except WebSocketDisconnect:
        return


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming voice interaction with low-latency streaming."""
    await websocket.accept()
    pipeline = get_pipeline()
    start_time = time.time()
    
    try:
        while True:
            # Receive audio frame or command with timeout for responsiveness
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                continue
            
            if data.get("type") == "state":
                # Send current state - fast response
                state = pipeline.get_state_dict()
                await websocket.send_json({
                    "type": "state",
                    "payload": state,
                    "timestamp": time.time()
                })
            
            elif data.get("type") == "transcription":
                # User transcription received - stream response chunks
                user_message = data.get("text", "").strip()
                if not user_message:
                    continue
                
                request_start = time.time()
                chat_request = ChatRequest(message=user_message, session_id="voice-session")
                
                cached = request_cache.get(chat_request)
                if cached:
                    response = cached
                    llm_time = 0.0
                else:
                    response = await agent.run(chat_request)
                    llm_time = time.time() - request_start
                    request_cache.set(chat_request, response)
                
                # Send immediate partial response while generating speech
                await websocket.send_json({
                    "type": "response_start",
                    "text": response.response[:100] + "..." if len(response.response) > 100 else response.response,
                    "llm_time_ms": int(llm_time * 1000),
                    "timestamp": time.time()
                })
                
                # Stream speech response asynchronously
                try:
                    clone_profile = data.get("voice_clone")
                    tts_start = time.time()
                    
                    # Non-blocking speech generation
                    await asyncio.to_thread(
                        pipeline.stream_speech,
                        response.response,
                        clone_profile
                    )
                    
                    tts_time = time.time() - tts_start
                    total_time = time.time() - request_start
                    
                    # Send complete response with metrics
                    await websocket.send_json({
                        "type": "response_complete",
                        "text": response.response,
                        "status": "complete",
                        "metrics": {
                            "llm_time_ms": int(llm_time * 1000),
                            "tts_time_ms": int(tts_time * 1000),
                            "total_time_ms": int(total_time * 1000)
                        },
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.error(f"TTS streaming error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Speech generation failed: {str(e)}"
                    })
            
            elif data.get("type") == "interrupt":
                # Stop current speech immediately
                try:
                    pipeline.interrupt_speech()
                    await websocket.send_json({
                        "type": "status",
                        "message": "Speech interrupted",
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.error(f"Interrupt error: {e}")
    
    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        logger.info(f"Voice WebSocket disconnected after {elapsed:.2f}s")
        await cleanup_pipeline()
        return
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"WebSocket error: {str(e)}",
                "timestamp": time.time()
            })
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BROWNIE_HOST", "0.0.0.0"),
        port=int(os.getenv("BROWNIE_PORT", "8000")),
        reload=os.getenv("BROWNIE_RELOAD", "false").lower() == "true",
        loop="asyncio",
    )
