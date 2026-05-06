# Brownie

Brownie is a proactive personal AI agent with a FastAPI backend, a LangGraph agent loop, persistent ChromaDB memory, an isolated Docker Python sandbox, and a minimal Next.js control surface.

## Architecture

The backend lives in `backend/main.py`.

- `FastAPI` exposes `GET /health`, `POST /chat`, `WS /ws/chat`, and workflow endpoints under `/workflows`.
- `LangGraph` controls the agent state loop: `load_memory -> decide -> run_tool|talk -> persist_memory`.
- `ChromaDB` persists completed actions and outcomes in a vector collection, using a local hash embedding function so the system can boot without external embedding services.
- `WorkflowStore` persists taught workflows in `workflows.json` beside the Chroma data, then injects matching workflows into Brownie's prompt context.
- `DockerSandbox` runs Python in a locked-down container with no network, CPU/memory limits, a read-only filesystem, and a timeout.
- `OPENAI_API_KEY` enables hosted LLM routing and response generation. `OPENAI_BASE_URL` can point Brownie at OpenAI-compatible providers such as Ollama, Groq, or OpenRouter. Without either, Brownie still runs with deterministic fallback routing.

The frontend lives in `frontend/`.

- `Next.js + TypeScript` provides the Brownie console.
- Local shadcn/ui primitives are checked into `src/components/ui`.
- The UI connects to `WS /ws/chat` and streams trace events before the final response.
- Browser voice input uses the Web Speech API when available, and voice output uses local `speechSynthesis`.
- Face enrollment/verification uses the browser camera and stores a lightweight local profile in `localStorage`; it is a convenience identity check, not a security-grade biometric system.
- Teach mode saves reusable trigger phrases and step lists through the backend workflow API.
- The visible reasoning panel shows operational trace summaries, not hidden model chain-of-thought.

## Run With Docker

Start Docker Desktop with Linux containers, then run:

```powershell
docker compose up --build
```

Open:

- Frontend: `http://localhost:3002`
- Backend health: `http://localhost:8010/health`

The compose defaults use `3002` and `8010` to avoid common local port conflicts. Override them when needed:

```powershell
$env:BROWNIE_FRONTEND_PORT="3000"
$env:BROWNIE_BACKEND_PORT="8000"
docker compose up --build
```

Optional LLM configuration:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:BROWNIE_MODEL="gpt-4o-mini"
docker compose up --build
```

Optional local Ollama configuration:

```powershell
ollama pull llama3.2
$env:OPENAI_API_KEY=""
$env:OPENAI_BASE_URL="http://host.docker.internal:11434/v1"
$env:BROWNIE_MODEL="llama3.2"
docker compose up --build
```

Optional OpenAI-compatible hosted provider configuration:

```powershell
$env:OPENAI_API_KEY="provider-key"
$env:OPENAI_BASE_URL="https://provider-openai-compatible-url/v1"
$env:BROWNIE_MODEL="provider-model-name"
docker compose up --build
```

The compose file mounts `/var/run/docker.sock` into the backend so the sandbox tool can launch isolated Python containers. Keep that mount restricted to trusted local development.

## Voice, Face, And Teach Mode

Voice:

1. Open `http://localhost:3002`.
2. Click the microphone button.
3. Allow browser microphone permission.
4. Speak a message. Brownie sends it and reads the reply if speaker output is enabled.

Face:

1. Click the camera button.
2. Allow browser camera permission.
3. Click the shield/check button to enroll.
4. Click verify on later visits to compare the current camera frame with the local profile.

Teach:

1. Add a workflow name.
2. Add a trigger phrase such as `daily startup`.
3. Add one step per line.
4. Save it, then click run or type the trigger phrase in chat.

Workflow API:

```powershell
Invoke-RestMethod "http://localhost:8010/workflows?session_id=brownie-web"
```

## Run Services Manually

Backend:

```powershell
docker build -t brownie-backend:phase1 ./backend
docker run --rm -p 8000:8000 -v /var/run/docker.sock:/var/run/docker.sock brownie-backend:phase1
```

Frontend:

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_BACKEND_WS_URL="ws://localhost:8000/ws/chat"
npm run dev
```

## Adding A Tool

Tools are routed from the LangGraph `decide` node in `backend/main.py`.

1. Add a tool class or function near `DockerSandbox`.
2. Register an instance beside `memory`, `sandbox`, and `agent`.
3. Update `_llm_decision` so the router can select the new tool name and JSON input shape.
4. Update `_fallback_decision` if the tool should work without an LLM.
5. Extend `run_tool` with a branch for the new tool.
6. Keep tool outputs shaped like `{ ok, stdout, stderr, exit_code }` or document a new shape before using it in `_llm_response`.

Every completed task is saved by `persist_memory`, so new tool outcomes automatically become part of Brownie's long-term memory.
