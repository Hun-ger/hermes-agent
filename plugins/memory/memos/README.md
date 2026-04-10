# MemOS Cloud Memory Plugin

A memory plugin for Hermes Agent using the [MemOS Cloud](https://memos.memtensor.cn) API. It provides server-side LLM fact extraction and semantic search.

## Configuration

Set the following environment variables or add them to `$HERMES_HOME/memos.json`:

- `MEMOS_API_KEY` (Required): Your MemOS API key.
- `MEMOS_BASE_URL` (Optional): The base URL for the MemOS API. Default: `https://memos.memtensor.cn/api/openmem/v1`.
- `MEMOS_USER_ID` (Optional): The user identifier. Default: `openclaw-user`.
- `MEMOS_AGENT_ID` (Optional): The agent identifier. Default: `hermes`.

## Features

- **Auto-sync**: Automatically saves your conversation turns to the MemOS Cloud.
- **Prefetch**: Automatically retrieves relevant memories before each conversation turn based on the user's input.
- **Tool**: Exposes the `memos_search` tool for semantic search over memories.

## How it works

This plugin is adapted from the MemOS Cloud OpenClaw plugin and implements the `MemoryProvider` interface for Hermes Agent. It calls the `/add/message` endpoint to persist conversations and `/search/memory` to retrieve context.
