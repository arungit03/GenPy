# Phase 12: GenPy Chat Interface

Phase 12 introduces a standalone Vite frontend in `frontend/` for chatting with
the existing GenPy FastAPI API.

## Features

- ChatGPT-style responsive layout with desktop sidebar and mobile drawer.
- Conversation history persisted in `localStorage`.
- New, rename, delete, select, and search chat actions.
- Markdown rendering with GitHub-flavored Markdown support.
- Syntax-highlighted code blocks with copy buttons.
- Message copy controls, typing indicator, auto-scroll, retry, regenerate, and
  stop generation controls.
- Settings panel for temperature, top-p, and max token generation.
- Model information panel backed by `/health` and `/model`.
- Light, dark, and system theme preferences.
- Keyboard shortcuts for new chat, search, settings, and escape-to-close.
- Axios error normalization for validation, connection, timeout, and server
  errors.
- Framer Motion transitions for messages, sidebars, and panels.

## API Contract

The UI calls the existing endpoints only:

```text
GET /health
GET /model
POST /generate
POST /chat
```

`POST /chat` sends messages in the backend schema:

```json
{
  "messages": [
    { "role": "user", "content": "Write a binary search function." }
  ],
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## Running Locally

Start the API in one terminal:

```bash
python scripts/run_api.py
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies API routes to `http://localhost:8000`. To point at another local
API server:

```bash
VITE_DEV_API_TARGET=http://localhost:9000 npm run dev
```

For production builds with a remote API origin:

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

## Verification

Expected frontend checks:

```bash
cd frontend
npm run lint
npm run typecheck
npm run build
```
