# GenPy Chat Interface

Phase 12 adds a React 19 chat frontend for the local GenPy FastAPI server.

## Commands

```bash
npm install
npm run dev
npm run lint
npm run typecheck
npm run build
```

The Vite dev server proxies `/health`, `/model`, `/generate`, and `/chat` to
`http://localhost:8000` by default. Override the backend target with:

```bash
VITE_DEV_API_TARGET=http://localhost:8000 npm run dev
```

For deployments where the frontend and backend are hosted on different origins,
set:

```bash
VITE_API_BASE_URL=https://your-api-host.example.com npm run build
```

## Structure

```text
src/
  api/          Axios client and API error normalization
  components/   Reusable chat, sidebar, panel, markdown, and input components
  store/        Zustand store with localStorage persistence
  types/        API and chat domain types
  utils/        Formatting and ID helpers
```

## Backend

The frontend uses the existing FastAPI contract and does not require backend
changes:

- `GET /health`
- `GET /model`
- `POST /generate`
- `POST /chat`
