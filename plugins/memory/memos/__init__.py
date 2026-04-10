"""MemOS Cloud memory plugin — Server-side LLM fact extraction and semantic search via MemOS API.

Config via environment variables:
  MEMOS_API_KEY      — MemOS API key (required)
  MEMOS_BASE_URL     — MemOS API base URL (default: https://memos.memtensor.cn/api/openmem/v1)
  MEMOS_USER_ID      — User identifier (default: openclaw-user)
  MEMOS_AGENT_ID     — Agent identifier (default: hermes)

Or via $HERMES_HOME/memos.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

import requests

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

def _load_config() -> dict:
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEMOS_API_KEY", ""),
        "base_url": os.environ.get("MEMOS_BASE_URL", "https://memos.memtensor.cn/api/openmem/v1").rstrip("/"),
        "user_id": os.environ.get("MEMOS_USER_ID", "openclaw-user"),
        "agent_id": os.environ.get("MEMOS_AGENT_ID", "hermes"),
        "source": "openclaw",
    }

    config_path = get_hermes_home() / "memos.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass

    return config

SEARCH_SCHEMA = {
    "name": "memos_search",
    "description": "Search memories by meaning via MemOS Cloud.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Max results (default: 5)."},
        },
        "required": ["query"],
    },
}

class MemosMemoryProvider(MemoryProvider):
    """MemOS Cloud memory provider."""

    def __init__(self):
        self._config = None
        self._api_key = ""
        self._base_url = ""
        self._user_id = ""
        self._agent_id = ""
        self._source = ""
        
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "memos"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key", "description": "MemOS API key", "secret": True, "required": True, "env_var": "MEMOS_API_KEY"},
            {"key": "base_url", "description": "MemOS API Base URL", "default": "https://memos.memtensor.cn/api/openmem/v1", "env_var": "MEMOS_BASE_URL"},
            {"key": "user_id", "description": "User identifier", "default": "openclaw-user", "env_var": "MEMOS_USER_ID"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes", "env_var": "MEMOS_AGENT_ID"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._base_url = self._config.get("base_url", "")
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "openclaw-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._source = self._config.get("source", "openclaw")

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "MemOS circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _call_api(self, path: str, payload: dict) -> Any:
        url = f"{self._base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self._api_key}"
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract data like extractResultData in JS
        if isinstance(data, dict):
            if "data" in data:
                inner = data["data"]
                if isinstance(inner, dict):
                    if "data" in inner:
                        return inner["data"]
                    if "result" in inner:
                        return inner["result"]
                return inner
            return data
        return data

    def system_prompt_block(self) -> str:
        return f"# MemOS Memory\nActive. User: {self._user_id}."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## MemOS Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open() or not query:
            return

        def _run():
            try:
                payload = {
                    "user_id": self._user_id,
                    "query": query,
                    "source": self._source,
                    "agent_id": self._agent_id,
                    "memory_limit_number": 5
                }
                if session_id:
                    payload["conversation_id"] = session_id
                    
                results = self._call_api("/search/memory", payload)
                if results and isinstance(results, list):
                    # Guessing format based on typical memory APIs
                    lines = []
                    for r in results:
                        if isinstance(r, dict):
                            content = r.get("content") or r.get("text") or r.get("memory")
                            if content:
                                lines.append(content)
                        elif isinstance(r, str):
                            lines.append(r)
                            
                    if lines:
                        with self._prefetch_lock:
                            self._prefetch_result = "\\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("MemOS prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="memos-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._is_breaker_open() or not user_content:
            return

        def _sync():
            try:
                messages = [{"role": "user", "content": user_content}]
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                    
                payload = {
                    "user_id": self._user_id,
                    "messages": messages,
                    "source": self._source,
                    "agent_id": self._agent_id
                }
                if session_id:
                    payload["conversation_id"] = session_id
                    
                self._call_api("/add/message", payload)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("MemOS sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="memos-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({"error": "MemOS API temporarily unavailable."})

        if tool_name == "memos_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            limit = min(int(args.get("limit", 5)), 20)
            
            try:
                payload = {
                    "user_id": self._user_id,
                    "query": query,
                    "source": self._source,
                    "agent_id": self._agent_id,
                    "memory_limit_number": limit
                }
                results = self._call_api("/search/memory", payload)
                self._record_success()
                
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                
                items = []
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict):
                            content = r.get("content") or r.get("text") or r.get("memory") or str(r)
                            items.append({"memory": content})
                        else:
                            items.append({"memory": str(r)})
                else:
                    items.append({"memory": str(results)})
                    
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

def register(ctx) -> None:
    ctx.register_memory_provider(MemosMemoryProvider())
