"""Thread-safe registry for managing user-defined functions.

This module provides a thread-safe container for function definitions, replacing
unsafe global dictionaries. It ensures that concurrent access (e.g., from multiple
worker threads or the REPL) does not corrupt the internal state.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """Thread-safe registry for user-defined functions.
    
    Stores function definitions as (params, body) tuples keyed by name.
    Uses RLock to safe-guard all read/write operations.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # Storage: name -> (params, body)
        self._data: Dict[str, Tuple[List[str], Any]] = {}

    def __getstate__(self):
        """Exclude lock from pickling to fix multiprocessing crash."""
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        """Restore state and recreate lock."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def register(self, name: str, params: list[str], body: Any) -> None:
        """Register or update a function definition.

        Args:
            name: Function name
            params: List of parameter names
            body: Function body (SymPy expression or supported object)
        """
        with self._lock:
            self._data[name] = (params, body)
            logger.debug(f"Registered function: {name}({', '.join(params)})")

    def get(self, name: str) -> Tuple[List[str], Any] | None:
        """Retrieve a function definition by name.

        Args:
            name: Function name to retrieve

        Returns:
            Tuple of (params, body) if found, else None
        """
        with self._lock:
            return self._data.get(name)

    def delete(self, name: str) -> bool:
        """Delete a function by name.

        Args:
            name: Function name to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if name in self._data:
                del self._data[name]
                logger.debug(f"Deleted function: {name}")
                return True
            return False

    def clear(self) -> None:
        """Remove all registered functions."""
        with self._lock:
            self._data.clear()
            logger.info("Cleared all user functions from registry.")

    def list_names(self) -> List[str]:
        """Get list of all registered function names."""
        with self._lock:
            return list(self._data.keys())

    def __getitem__(self, name: str) -> Tuple[List[str], Any]:
        with self._lock:
            return self._data[name]

    def __setitem__(self, name: str, value: Tuple[List[str], Any]) -> None:
        with self._lock:
            self._data[name] = value

    def __delitem__(self, name: str) -> None:
        with self._lock:
            del self._data[name]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._data

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def items(self) -> List[Tuple[str, Tuple[List[str], Any]]]:
        """Get all items as a list of (name, (params, body))."""
        with self._lock:
            return list(self._data.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
