# AdapterAI 🚀

> Multi-agent AI platform with a FastAPI backend, sandboxed tool-executor, PostgreSQL database, and Redis cache — all containerised and ready to run with **one command**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      adapterai_net                          │
│                                                             │
│  ┌───────────┐   ┌───────────┐   ┌──────────┐  ┌────────┐  │
│  │  backend  │──▶│ executor  │   │ postgres │  │ redis  │  │
│  │ :8002     │   │ :8000     │   │ :5432    │  │ :6379  │  │
│  └───────────┘   └───────────┘   └──────────┘  └────────┘  │
│        │                              ▲              ▲      │
│        └──────────────────────────────┴──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

| Container | Image / Build | Purpose | Ports |
|---|---|---|---|
| `adapterai_backend` | `./Dockerfile` | FastAPI app (all agents + REST API) | **8002** |
| `adapterai_executor` | `./Docker_exec/Dockerfile` | Sandboxed Python tool execution | **8000** |
| `adapterai_postgres` | `postgres:15` | Primary relational database | **5432** |
| `adapterai_redis` | `redis/redis-stack` | Cache, checkpointing & message queue | **6379** |

**Bonus UIs (started automatically):**
- **pgAdmin** → http://localhost:5050 (DB browser)
- **RedisInsight** → http://localhost:8001 (Redis browser)

---

## Prerequisites

| Tool | Minimum Version | Check |
|---|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 24.x | `docker --version` |
| [Docker Compose](https://docs.docker.com/compose/) | v2.x (bundled) | `docker compose version` |

> Docker Desktop must be running before executing any command below.

---

## Quickstart

### 1 — Clone and enter the project

```bash
git clone <your-repo-url> AdapterAI
cd AdapterAI
```

### 2 — Create your `.env` file

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your real API keys (see [Environment Variables](#environment-variables) below).

### 3 — Start everything

```bash
docker compose up --build
```

That's it. Docker will:
1. Build the `backend` image from the root `Dockerfile`
2. Build the `executor` image from `Docker_exec/Dockerfile`
3. Pull `postgres:15` and `redis/redis-stack`
4. Run all 4 containers on a shared private network
5. Apply `Docker_postgresql/init.sql` to initialise the database schema

> First build takes ~3–5 minutes (dependency download). Subsequent starts are instant.

### 4 — Verify everything is healthy

```bash
docker compose ps
```

All four services should show **healthy** status.

```bash
# Quick smoke-test
curl http://localhost:8002/          # {"status":"ok","service":"AdapterAI API"}
curl http://localhost:8000/health    # {"status":"ok","service":"tool-executor"}
```

---

## Useful Commands

| Task | Command |
|---|---|
| Start (foreground) | `docker compose up --build` |
| Start (background) | `docker compose up --build -d` |
| Stop | `docker compose down` |
| Stop + wipe volumes | `docker compose down -v` |
| View logs (all) | `docker compose logs -f` |
| View backend logs | `docker compose logs -f backend` |
| Rebuild one service | `docker compose up --build backend` |
| Open PostgreSQL shell | `docker exec -it adapterai_postgres psql -U postgres -d app_db` |
| Open Redis CLI | `docker exec -it adapterai_redis redis-cli` |

---

## Service URLs

| Service | URL | Description |
|---|---|---|
| **Backend API** | http://localhost:8002 | Health check |
| **Swagger UI** | http://localhost:8002/docs | Interactive API docs |
| **ReDoc** | http://localhost:8002/redoc | Alternative docs |
| **Tool Executor** | http://localhost:8000/health | Executor health |
| **pgAdmin** | http://localhost:5050 | Database browser |
| **RedisInsight** | http://localhost:8001 | Redis browser |
| **PostgreSQL** | localhost:5432 | Direct DB connection |
| **Redis** | localhost:6379 | Direct cache connection |

---

## Environment Variables

Copy `.env.example` to `.env` and set the values below.

### Required (app won't start without these)

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq LLM API key |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `TAVILY_API_KEY` | Tavily web-search API key |
| `CLOUDFLARE_API_TOKEN` | Cloudflare Workers AI (embeddings) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service-role key |
| `PINECONE_API_KEY` | Pinecone vector-store API key |
| `JWT_SECRET_KEY` | Secret for signing JWTs (use a long random string) |

### Optional (have sensible defaults)

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_USER` | `postgres` | DB username |
| `POSTGRES_PASSWORD` | `password` | DB password |
| `POSTGRES_DB` | `app_db` | Database name |
| `POSTGRES_PORT` | `5432` | Host-side port |
| `REDIS_PORT` | `6379` | Host-side Redis port |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINS` | `60` | Token lifetime in minutes |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith traces |
| `LANGSMITH_API_KEY` | — | LangSmith key (if tracing enabled) |

> Never commit your real `.env` file. It is already listed in `.gitignore`.

---

## Project Structure

```
AdapterAI/
├── Dockerfile                  <- Backend image (built from root)
├── docker-compose.yml          <- Unified compose (all 4 services)
├── .env.example                <- Environment variable template
├── .dockerignore               <- Excludes venv, cache, etc.
├── requirements.txt            <- Combined Python deps for backend
│
├── apis/                       <- FastAPI entry point & routers
├── MainAgent/                  <- LangGraph orchestrator agent
├── SubAgent/                   <- Custom-tool sub-agents
├── ToolGeneration/             <- AI-powered tool generator
├── TemplateCreation/           <- Template management
├── builtintools/               <- Web search, retrieval tools
├── vector_store/               <- Pinecone embedding pipeline
├── utils/                      <- Redis checkpointer, tracing
│
├── Docker_exec/                <- Tool executor service
│   ├── Dockerfile
│   ├── api.py
│   ├── runner.py
│   └── tool_compiler.py
│
├── Docker_postgresql/          <- PostgreSQL init scripts
│   └── init.sql
│
└── Docker_redis/               <- (legacy standalone compose)
```

---

## Troubleshooting

### Port conflict
If a port is already in use, change it in `.env`:
```bash
POSTGRES_PORT=5433
REDIS_PORT=6380
```
Then restart:
```bash
docker compose down && docker compose up --build
```

### Backend can't reach postgres / redis
The backend container automatically overrides `POSTGRES_DSN` and `REDIS_URL` with Docker service hostnames (`postgres`, `redis`) in `docker-compose.yml`. Do **not** manually set `localhost` for these.

### Database schema missing
The schema is applied from `Docker_postgresql/init.sql` only on the first boot (when the volume is empty). To re-apply from scratch:
```bash
docker compose down -v   # WARNING: deletes all data
docker compose up --build
```

### Rebuilding after code changes
```bash
docker compose up --build backend   # rebuild only the backend image
```

---

## License

MIT
