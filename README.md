# HCP Clarifier

[![Python](https://img.shields.io/badge/Python-3.9-3776AB)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-DeepSeek-1C3C3C)](https://www.langchain.com/)

A query-clarification agent that turns a vague user question into a precise one through targeted follow-up, before it's handed off to a downstream system. Built with LangChain and the DeepSeek chat model.

## How it works

For each query, the clarifier loops (up to a configurable max round count):

1. **Classify** — decide whether the current query is already clear enough to act on (`SIMPLE`) or needs clarification.
2. **Strategize** — pick what's missing: user intent, background context, or specific details.
3. **Ask** — generate one targeted follow-up question for that gap.
4. **Refine** — fold the user's answer back into the query and repeat.

The loop stops early once the query is classified as clear, or after the max round count is reached, and produces a final, refined query either way.

## Interfaces

- **CLI** (`claifier_bot.py`) — an interactive terminal loop for testing the clarifier directly.
- **HTTP API** (`api_service.py`, Flask) — stateful multi-turn clarification over HTTP:
  - `POST /clarify/start` — start a clarification session for a new query.
  - `POST /clarify/continue` — continue an existing session with the user's answer.
  - `GET /health` — health check.

## Setup

```bash
pip install -r requirement.txt
```

Create a `.env` file with your DeepSeek API key:

```dotenv
DEEPSEEK_API_KEY=your-key-here
```

Run the CLI:

```bash
python claifier_bot.py
```

Or the API server:

```bash
python api_service.py
```

## Status

Early-stage / experimental. Note for the repo owner: `.env` (with a placeholder key) and the `__pycache__/` directory are currently tracked in git — worth adding both to `.gitignore` and untracking them so a real API key never accidentally ends up in a commit.
