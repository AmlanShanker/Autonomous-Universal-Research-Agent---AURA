# AURA Client

React + TypeScript + Vite frontend for the AURA FastAPI backend.

## Requirements

- Node.js 20.19+ or 22.12+
- npm
- AURA API running at `http://127.0.0.1:8000` (or configure another URL)

## Run locally (Windows PowerShell)

From the repository root:

```powershell
cd client
Copy-Item .env.example .env
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

In a separate terminal, start the backend from the repository root:

```powershell
uv run uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

## Configuration

Set `VITE_AURA_API_URL` in `.env` to the base URL of the AURA API. The default is `http://127.0.0.1:8000`.

Do not put provider API keys, MongoDB credentials, or other backend secrets in Vite environment variables. Values prefixed with `VITE_` are bundled into browser code.

## Current scope

- Responsive AURA research workspace shell.
- Backend health status via `GET /health`.
- Research run creation via `POST /research/runs`.
- Displays the returned research plan and task graph.

Research execution, run history persistence, authentication UX, and report browsing are intentionally left for follow-up frontend milestones.
