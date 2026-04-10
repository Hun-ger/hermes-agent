# MemOS Memory Provider

Server-side LLM fact extraction and semantic search via the MemOS Cloud API.

## Requirements

- `pip install requests` (Handled automatically)
- MemOS API key from [MemOS Cloud](https://memos.memtensor.cn)

## Setup

```bash
hermes memory setup    # select "memos"
```

Or manually:
```bash
hermes config set memory.provider memos
echo "MEMOS_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/memos.json`

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `https://memos.memtensor.cn/api/openmem/v1` | MemOS API base URL |
| `user_id` | `openclaw-user` | User identifier on MemOS |
| `agent_id` | `hermes` | Agent identifier |

## Tools

| Tool | Description |
|------|-------------|
| `memos_search` | Semantic search over memories |
