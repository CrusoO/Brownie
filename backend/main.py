import asyncio
import hashlib
import json
import os
import re
import subprocess
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

import chromadb
from chromadb.api.types import Documents, Embeddings
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - the fallback brain keeps the API runnable.
    HumanMessage = None
    SystemMessage = None
    ChatOpenAI = None

from langgraph.graph import END, START, StateGraph


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
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    openai_model: str = Field(default_factory=lambda: os.getenv("BROWNIE_MODEL", "gpt-4o-mini"))
    sandbox_image: str = Field(default_factory=lambda: os.getenv("BROWNIE_SANDBOX_IMAGE", "python:3.12-alpine"))
    sandbox_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("BROWNIE_SANDBOX_TIMEOUT_SECONDS", "8"))
    )
    max_python_chars: int = Field(default_factory=lambda: int(os.getenv("BROWNIE_MAX_PYTHON_CHARS", "12000")))


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


class BrownieState(TypedDict, total=False):
    run_id: str
    session_id: str
    user_message: str
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
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.workflows = workflows
        self.sandbox = sandbox
        self.llm = self._build_llm()
        self.graph = self._build_graph()

    def _build_llm(self) -> Any:
        api_key = self.settings.openai_api_key
        if not api_key and self.settings.openai_base_url:
            api_key = "ollama"

        if not api_key or ChatOpenAI is None:
            return None

        kwargs: dict[str, Any] = {
            "model": self.settings.openai_model,
            "api_key": api_key,
            "temperature": 0.2,
        }
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url

        return ChatOpenAI(**kwargs)

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
        memories = self.memory.search(state["user_message"], state["session_id"])
        workflow_matches = self.workflows.find_matches(state["user_message"], state["session_id"])
        workflow_context = [self.workflows.format_for_prompt(workflow) for workflow in workflow_matches]
        event = await self._event(
            state,
            "memory.loaded",
            "Checked long-term memory.",
            {"matches": len(memories), "workflows": len(workflow_context)},
        )
        return {
            "memory_context": memories,
            "workflow_context": workflow_context,
            "trace": [*state.get("trace", []), event],
        }

    async def decide(self, state: BrownieState) -> BrownieState:
        llm_error = None
        if self.llm:
            try:
                decision = await self._llm_decision(state)
            except Exception as exc:
                llm_error = self._format_llm_error(exc)
                decision = self._fallback_decision(state["user_message"])
        else:
            decision = self._fallback_decision(state["user_message"])

        event_data = {
            "route": decision["route"],
            "tool_name": decision.get("tool_name"),
        }
        if llm_error:
            event_data["llm_error"] = llm_error

        event = await self._event(
            state,
            "agent.route",
            "Chose the next action.",
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
        if tool_name != "python_sandbox":
            result = {
                "ok": False,
                "stdout": "",
                "stderr": f"Unknown tool: {tool_name}",
                "exit_code": 1,
            }
        else:
            start_event = await self._event(
                state,
                "tool.started",
                "Running Python inside the Docker sandbox.",
                {"tool_name": "python_sandbox"},
            )
            state = {**state, "trace": [*state.get("trace", []), start_event]}
            result = await self.sandbox.run_python(str(tool_input.get("code", "")))

        finish_event = await self._event(
            state,
            "tool.finished",
            "Tool execution finished.",
            {
                "tool_name": tool_name,
                "ok": result["ok"],
                "exit_code": result["exit_code"],
            },
        )
        return {
            "tool_result": result,
            "trace": [*state.get("trace", []), finish_event],
        }

    async def talk(self, state: BrownieState) -> BrownieState:
        llm_error = state.get("llm_error")
        if self.llm and not llm_error:
            try:
                response = await self._llm_response(state)
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
        memory_id = self.memory.remember(
            session_id=state["session_id"],
            action=state["user_message"],
            outcome=state.get("response", ""),
            route=state.get("route", "talk"),
            tool_name=state.get("tool_name"),
        )
        event = await self._event(
            state,
            "memory.saved",
            "Saved this action and outcome to long-term memory.",
            {"memory_id": memory_id},
        )
        return {
            "memory_ids": [*state.get("memory_ids", []), memory_id],
            "trace": [*state.get("trace", []), event],
        }

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

    async def _llm_decision(self, state: BrownieState) -> dict[str, Any]:
        if HumanMessage is None or SystemMessage is None:
            return self._fallback_decision(state["user_message"])

        system = (
            "You are Brownie's router. Return only compact JSON. "
            "Choose route='tool' only when the user explicitly asks to run Python code or provides code to execute. "
            "If a taught workflow is present, route='talk' unless the user's message or workflow contains literal Python code to run. "
            "The only available tool is python_sandbox. "
            "For tool use, return: {\"route\":\"tool\",\"tool_name\":\"python_sandbox\",\"tool_input\":{\"code\":\"...\"}}. "
            "Otherwise return: {\"route\":\"talk\",\"tool_name\":null,\"tool_input\":{}}."
        )
        context = "\n\n".join(state.get("memory_context", [])) or "No relevant memory."
        workflows = "\n\n".join(state.get("workflow_context", [])) or "No matching taught workflows."
        raw = await self.llm.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        f"Relevant memory:\n{context}\n\n"
                        f"Taught workflows:\n{workflows}\n\n"
                        f"User message:\n{state['user_message']}"
                    )
                ),
            ]
        )
        return self._normalize_decision(self._json_from_text(raw.content), state["user_message"])

    async def _llm_response(self, state: BrownieState) -> str:
        if HumanMessage is None or SystemMessage is None:
            return self._fallback_response(state)

        memory = "\n\n".join(state.get("memory_context", [])) or "No relevant memory."
        workflows = "\n\n".join(state.get("workflow_context", [])) or "No matching taught workflows."
        tool_result = state.get("tool_result")
        tool_summary = json.dumps(tool_result, ensure_ascii=True) if tool_result else "No tool was used."
        system = (
            "You are Brownie, a proactive personal AI agent. "
            "Use the visible memory and tool result to answer clearly. "
            "When a taught workflow matches the user request, execute the workflow conversationally: "
            "state what you can do now, use any available tool result, and ask for confirmation only for unsafe or external actions. "
            "Do not claim you changed files, opened apps, sent messages, or contacted services unless a tool result proves it. "
            "Do not reveal hidden chain-of-thought; summarize actions and outcomes instead."
        )
        raw = await self.llm.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        f"Relevant memory:\n{memory}\n\n"
                        f"Taught workflows:\n{workflows}\n\n"
                        f"Tool result:\n{tool_summary}\n\n"
                        f"User message:\n{state['user_message']}"
                    )
                ),
            ]
        )
        return str(raw.content).strip()

    def _fallback_decision(self, message: str) -> dict[str, Any]:
        code = self._extract_python(message)
        if code:
            return {
                "route": "tool",
                "tool_name": "python_sandbox",
                "tool_input": {"code": code},
            }

        return {
            "route": "talk",
            "tool_name": None,
            "tool_input": {},
        }

    def _fallback_response(self, state: BrownieState, llm_error: Optional[str] = None) -> str:
        tool_result = state.get("tool_result")
        if tool_result:
            stdout = (tool_result.get("stdout") or "").strip()
            stderr = (tool_result.get("stderr") or "").strip()
            if tool_result.get("ok"):
                return stdout or "The sandboxed Python code completed successfully with no stdout."
            return f"Sandbox execution failed with exit code {tool_result.get('exit_code')}.\n\n{stderr or stdout}"

        if llm_error:
            return (
                f"OpenAI request failed: {llm_error}\n\n"
                "Brownie is still running, but the language-model call could not complete. "
                "Check your provider billing/quota, or remove OPENAI_API_KEY and OPENAI_BASE_URL from `.env` "
                "to use fallback mode."
            )

        if state.get("workflow_context"):
            return (
                "I found a taught workflow that matches this request.\n\n"
                f"{state['workflow_context'][0]}"
            )

        memory_hint = ""
        if state.get("memory_context"):
            memory_hint = "\n\nI found related long-term memory and will keep using it as the agent gets smarter."

        return (
            "Brownie backend is running. Configure OPENAI_API_KEY or OPENAI_BASE_URL to enable full "
            "language-model responses; "
            "the LangGraph loop, memory layer, and Docker sandbox are ready."
            f"{memory_hint}"
        )

    def _format_llm_error(self, exc: Exception) -> str:
        text = str(exc)
        lower_text = text.lower()
        if "insufficient_quota" in lower_text or "exceeded your current quota" in lower_text:
            return "insufficient OpenAI API quota; add billing/credits or use a different API project."
        if "rate limit" in lower_text or exc.__class__.__name__ == "RateLimitError":
            return "OpenAI rate limit reached; wait briefly and try again."
        if "authentication" in lower_text or "api key" in lower_text:
            return "model provider authentication failed; check OPENAI_API_KEY and OPENAI_BASE_URL in `.env`."
        return f"{exc.__class__.__name__}: {text[:240]}"

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
agent = BrownieAgent(settings, memory, workflows, sandbox)

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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await agent.run(request)


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


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            request = ChatRequest(**payload)

            async def emit(event: dict[str, Any]) -> None:
                await websocket.send_json({"type": "trace", "event": event})

            response = await agent.run(request, emit=emit)
            await websocket.send_json({"type": "final", "response": response.model_dump()})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BROWNIE_HOST", "0.0.0.0"),
        port=int(os.getenv("BROWNIE_PORT", "8000")),
        reload=os.getenv("BROWNIE_RELOAD", "false").lower() == "true",
        loop="asyncio",
    )
