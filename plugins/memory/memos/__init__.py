"""MemOS memory plugin — MemoryProvider interface.

Server-side LLM fact extraction and semantic search via the MemOS Cloud API.

Config via environment variables:
  MEMOS_API_KEY      — MemOS Platform API key (required)
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

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/memos.json overrides."""
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


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "memos_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "and tool memories. Use at conversation start to understand the user."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

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

CONCLUDE_SCHEMA = {
    "name": "memos_conclude",
    "description": (
        "Store a durable fact or preference about the user manually. "
        "Stored verbatim (no LLM extraction)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

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
        
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "memos"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/memos.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "memos.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key", "description": "MemOS Platform API key", "secret": True, "required": True, "env_var": "MEMOS_API_KEY"},
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

    def _format_context_block(self, data: dict, max_item_chars: int = 1000) -> str:
        """Format the memory payload similar to the OpenClaw plugin's formatContextBlock."""
        if not data or not isinstance(data, dict):
            return ""

        memory_list = data.get("memory_detail_list", [])
        pref_list = data.get("preference_detail_list", [])
        tool_list = data.get("tool_memory_detail_list", [])
        preference_note = data.get("preference_note")

        lines = []
        if memory_list:
            lines.append("Facts:")
            for item in memory_list:
                text = item.get("memory_value") or item.get("memory_key") or ""
                if text:
                    lines.append(f"- {text[:max_item_chars]}")

        if pref_list:
            lines.append("Preferences:")
            for item in pref_list:
                pref = item.get("preference") or ""
                type_str = f"({item['preference_type']}) " if item.get("preference_type") else ""
                if pref:
                    lines.append(f"- {type_str}{pref[:max_item_chars]}")

        if tool_list:
            lines.append("Tool Memories:")
            for item in tool_list:
                value = item.get("tool_value") or ""
                if value:
                    lines.append(f"- {value[:max_item_chars]}")

        if preference_note:
            lines.append(f"Preference Note: {preference_note[:max_item_chars]}")

        return "\n".join(lines) if lines else ""

    def system_prompt_block(self) -> str:
        return (
            "# MemOS Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use memos_search to find memories, memos_conclude to store facts, "
            "memos_profile for a full overview."
        )

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
                if results and isinstance(results, dict):
                    formatted = self._format_context_block(results)
                    if formatted:
                        with self._prefetch_lock:
                            self._prefetch_result = formatted
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
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({"error": "MemOS API temporarily unavailable."})

        if tool_name == "memos_profile":
            try:
                payload = {
                    "user_id": self._user_id,
                    "query": "",
                    "source": self._source,
                    "agent_id": self._agent_id,
                    "memory_limit_number": 20,
                    "preference_limit_number": 10,
                    "tool_memory_limit_number": 10,
                    "include_preference": True,
                    "include_tool_memory": True
                }
                results = self._call_api("/search/memory", payload)
                self._record_success()
                
                formatted = self._format_context_block(results) if isinstance(results, dict) else ""
                if not formatted:
                    return json.dumps({"result": "No memories stored yet."})
                return json.dumps({"result": formatted})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "memos_search":
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
                    "memory_limit_number": limit,
                    "preference_limit_number": limit,
                    "tool_memory_limit_number": limit,
                    "include_preference": True,
                    "include_tool_memory": True
                }
                results = self._call_api("/search/memory", payload)
                self._record_success()
                
                formatted = self._format_context_block(results) if isinstance(results, dict) else ""
                if not formatted:
                    return json.dumps({"result": "No relevant memories found."})
                return json.dumps({"result": formatted})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "memos_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                payload = {
                    "user_id": self._user_id,
                    "messages": [{"role": "user", "content": conclusion}],
                    "source": self._source,
                    "agent_id": self._agent_id
                }
                self._call_api("/add/message", payload)
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

def register(ctx) -> None:
    ctx.register_memory_provider(MemosMemoryProvider())
