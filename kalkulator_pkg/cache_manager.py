"""Persistent cache management for Kalkulator.

This module provides:
- File-based persistent caching that survives process restarts
- Sub-expression caching (e.g., if "2+2" is cached as "4", use it in "(2+2)/2")
- Cache loading/saving with JSON storage
- Cache expiration and cleanup
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .logging_config import get_logger

    logger = get_logger("cache")
except ImportError:
    logger = None


# Cache file location
_CACHE_DIR = Path.home() / ".kalkulator_cache"
_CACHE_FILE = _CACHE_DIR / "cache.json"
_CACHE_VERSION = 1  # Increment when cache format changes


def _get_cache_dir() -> Path:
    """Get or create the cache directory."""
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR


def _get_cache_file() -> Path:
    """Get the cache file path."""
    return _CACHE_FILE


def load_persistent_cache() -> dict[str, Any]:
    """Load persistent cache from disk.

    Returns:
        Dictionary with cache data: {
            "version": int,
            "eval_cache": {preprocessed_expr: result_json},
            "subexpr_cache": {preprocessed_expr: result_value}
        }
    """
    cache_file = _get_cache_file()
    if not cache_file.exists():
        return {
            "version": _CACHE_VERSION,
            "eval_cache": {},
            "subexpr_cache": {},
        }

    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)

        # Check version compatibility
        if data.get("version") != _CACHE_VERSION:
            if logger:
                logger.info("Cache version mismatch, clearing old cache")
            return {
                "version": _CACHE_VERSION,
                "eval_cache": {},
                "subexpr_cache": {},
            }

        # Validate structure
        if not isinstance(data.get("eval_cache"), dict):
            data["eval_cache"] = {}
        if not isinstance(data.get("subexpr_cache"), dict):
            data["subexpr_cache"] = {}

        if logger:
            logger.debug(
                f"Loaded {len(data.get('eval_cache', {}))} eval cache entries and {len(data.get('subexpr_cache', {}))} subexpr cache entries"
            )

        return data
    except (json.JSONDecodeError, OSError) as e:
        if logger:
            logger.warning(f"Failed to load cache: {e}, starting with empty cache")
        return {
            "version": _CACHE_VERSION,
            "eval_cache": {},
            "subexpr_cache": {},
        }


def save_persistent_cache(cache_data: dict[str, Any]) -> None:
    """Save persistent cache to disk.

    Args:
        cache_data: Cache dictionary to save
    """
    cache_file = _get_cache_file()
    try:
        # Ensure cache directory exists
        _get_cache_dir()

        # Limit cache sizes before saving (keep most recent)
        # For eval_cache, limit to 5000 entries
        eval_cache = cache_data.get("eval_cache", {})
        if len(eval_cache) > 5000:
            # Keep only the last 5000 entries (LRU-like behavior)
            eval_cache = dict(list(eval_cache.items())[-5000:])
            cache_data["eval_cache"] = eval_cache

        # For subexpr_cache, limit to 10000 entries (sub-expressions are smaller)
        subexpr_cache = cache_data.get("subexpr_cache", {})
        if len(subexpr_cache) > 10000:
            subexpr_cache = dict(list(subexpr_cache.items())[-10000:])
            cache_data["subexpr_cache"] = subexpr_cache

        # Write atomically (write to temp file then rename)
        temp_file = cache_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        temp_file.replace(cache_file)

        if logger:
            logger.debug(
                f"Saved cache: {len(eval_cache)} eval, {len(subexpr_cache)} subexpr entries"
            )
    except (OSError, TypeError) as e:
        if logger:
            logger.warning(f"Failed to save cache: {e}")


# Global cache storage
_persistent_cache: dict[str, Any] | None = None

# Global cache hit tracking (last N hits for display)
_cache_hit_tracking: list[tuple[str, str]] = []  # [(expr, cache_type)]
_MAX_CACHE_HIT_TRACKING = 100  # Keep last 100 cache hits


def get_persistent_cache() -> dict[str, Any]:
    """Get or initialize the persistent cache."""
    global _persistent_cache
    if _persistent_cache is None:
        _persistent_cache = load_persistent_cache()
    return _persistent_cache


def update_eval_cache(
    preprocessed_expr: str,
    result_json: str,
    compute_time: float | None = None,
    context_hash: str | None = None,
) -> None:
    """Update the persistent evaluation cache.

    Args:
        preprocessed_expr: Preprocessed expression string
        result_json: JSON string of the evaluation result
        compute_time: Time taken to compute (in seconds), or None if not measured
        context_hash: Optional hash of the execution context (e.g., function registry)
    """
    cache = get_persistent_cache()
    # Store as dict with result and timing
    cache_entry = {"result": result_json}
    if compute_time is not None:
        cache_entry["time"] = compute_time

    cache_key = preprocessed_expr
    if context_hash:
        cache_key = f"{context_hash}:{preprocessed_expr}"

    cache["eval_cache"][cache_key] = cache_entry


def update_subexpr_cache(
    preprocessed_expr: str, result_value: str, compute_time: float | None = None
) -> None:
    """Update the persistent sub-expression cache.

    Args:
        preprocessed_expr: Preprocessed sub-expression string
        result_value: String representation of the cached value
        compute_time: Time taken to compute (in seconds), or None if not measured
    """
    cache = get_persistent_cache()
    # Store as dict with value and timing
    cache_entry = {"value": result_value}
    if compute_time is not None:
        cache_entry["time"] = compute_time
    cache["subexpr_cache"][preprocessed_expr] = cache_entry


def get_cached_eval(
    preprocessed_expr: str, context_hash: str | None = None
) -> str | None:
    """Get cached evaluation result if available.

    Args:
        preprocessed_expr: Preprocessed expression string
        context_hash: Optional hash of the execution context (e.g., function registry)

    Returns:
        Cached result JSON string, or None if not found
    """
    cache = get_persistent_cache()

    cache_key = preprocessed_expr
    if context_hash:
        cache_key = f"{context_hash}:{preprocessed_expr}"

    entry = cache["eval_cache"].get(cache_key)
    if entry is None:
        return None
    # Track cache hit
    _cache_hit_tracking.append((preprocessed_expr, "eval"))
    if len(_cache_hit_tracking) > _MAX_CACHE_HIT_TRACKING:
        _cache_hit_tracking.pop(0)  # Remove oldest
    # Support both old format (string) and new format (dict)
    if isinstance(entry, dict):
        res = entry.get("result")
        return str(res) if res is not None else None
    return str(entry) if entry is not None else None  # Old format


def get_cached_eval_with_time(
    preprocessed_expr: str, context_hash: str | None = None
) -> tuple[str | None, float | None]:
    """Get cached evaluation result with timing information.

    Args:
        preprocessed_expr: Preprocessed expression string
        context_hash: Optional hash of the execution context (e.g., function registry)

    Returns:
        Tuple of (cached result JSON string or None, compute time in seconds or None)
    """
    cache = get_persistent_cache()

    cache_key = preprocessed_expr
    if context_hash:
        cache_key = f"{context_hash}:{preprocessed_expr}"

    entry = cache["eval_cache"].get(cache_key)
    if entry is None:
        return None, None
    # Support both old format (string) and new format (dict)
    if isinstance(entry, dict):
        return entry.get("result"), entry.get("time")
    return entry, None  # Old format - no timing info


def get_cached_subexpr(preprocessed_expr: str) -> str | None:
    """Get cached sub-expression value if available.

    Args:
        preprocessed_expr: Preprocessed sub-expression string

    Returns:
        Cached value string, or None if not found
    """
    cache = get_persistent_cache()
    entry = cache["subexpr_cache"].get(preprocessed_expr)
    if entry is None:
        return None
    # Track cache hit
    _cache_hit_tracking.append((preprocessed_expr, "subexpr"))
    if len(_cache_hit_tracking) > _MAX_CACHE_HIT_TRACKING:
        _cache_hit_tracking.pop(0)  # Remove oldest
    # Support both old format (string) and new format (dict)
    if isinstance(entry, dict):
        return entry.get("value")
    return entry  # Old format - just return the string


def get_cached_subexpr_with_time(
    preprocessed_expr: str,
) -> tuple[str | None, float | None]:
    """Get cached sub-expression value with timing information.

    Args:
        preprocessed_expr: Preprocessed sub-expression string

    Returns:
        Tuple of (cached value string or None, compute time in seconds or None)
    """
    cache = get_persistent_cache()
    entry = cache["subexpr_cache"].get(preprocessed_expr)
    if entry is None:
        return None, None
    # Support both old format (string) and new format (dict)
    if isinstance(entry, dict):
        return entry.get("value"), entry.get("time")
    return entry, None  # Old format - no timing info


def get_cache_hits() -> list[tuple[str, str]]:
    """Get the list of recent cache hits.

    Returns:
        List of tuples (expression, cache_type) where cache_type is "eval" or "subexpr"
    """
    return _cache_hit_tracking.copy()


def clear_cache_hits() -> None:
    """Clear the cache hit tracking list."""
    global _cache_hit_tracking
    _cache_hit_tracking = []


def save_cache_to_disk() -> None:
    """Save the current cache state to disk."""
    if _persistent_cache is not None:
        save_persistent_cache(_persistent_cache)


def clear_persistent_cache() -> None:
    """Clear all persistent caches."""
    global _persistent_cache
    _persistent_cache = {
        "version": _CACHE_VERSION,
        "eval_cache": {},
        "subexpr_cache": {},
    }
    save_cache_to_disk()


def export_cache_to_file(file_path: str) -> bool:
    """Export the current cache to a JSON file.

    Args:
        file_path: Path to save the cache file

    Returns:
        True if successful, False otherwise
    """
    try:
        cache = get_persistent_cache()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        if logger:
            logger.debug(f"Exported cache to {file_path}")
        return True
    except (OSError, TypeError) as e:
        if logger:
            logger.warning(f"Failed to export cache: {e}")
        return False


def import_cache_from_file(file_path: str) -> bool:
    """Import cache from a JSON file and merge with existing cache.

    Args:
        file_path: Path to load the cache file from

    Returns:
        True if successful, False otherwise
    """
    global _persistent_cache
    try:
        with open(file_path, encoding="utf-8") as f:
            imported_data = json.load(f)

        # Validate structure
        if not isinstance(imported_data, dict):
            if logger:
                logger.warning("Imported cache file has invalid structure")
            return False

        # Get current cache
        current_cache = get_persistent_cache()

        # Merge imported cache with current cache
        # Imported entries take precedence (will overwrite existing)
        if isinstance(imported_data.get("eval_cache"), dict):
            current_cache["eval_cache"].update(imported_data["eval_cache"])

        if isinstance(imported_data.get("subexpr_cache"), dict):
            current_cache["subexpr_cache"].update(imported_data["subexpr_cache"])

        # Update the global cache
        _persistent_cache = current_cache

        # Save merged cache
        save_cache_to_disk()

        if logger:
            logger.debug(f"Imported and merged cache from {file_path}")
        return True
    except (OSError, json.JSONDecodeError) as e:
        if logger:
            logger.warning(f"Failed to import cache: {e}")
        return False


def replace_cache_from_file(file_path: str) -> bool:
    """Replace the entire cache with contents from a file.

    Args:
        file_path: Path to load the cache file from

    Returns:
        True if successful, False otherwise
    """
    global _persistent_cache
    try:
        with open(file_path, encoding="utf-8") as f:
            imported_data = json.load(f)

        # Validate structure and version
        if not isinstance(imported_data, dict):
            if logger:
                logger.warning("Imported cache file has invalid structure")
            return False

        # Check version compatibility
        if imported_data.get("version") != _CACHE_VERSION:
            if logger:
                logger.warning(
                    f"Cache version mismatch: expected {_CACHE_VERSION}, got {imported_data.get('version')}"
                )
            # Still allow import but warn

        # Validate cache structure
        if not isinstance(imported_data.get("eval_cache"), dict):
            imported_data["eval_cache"] = {}
        if not isinstance(imported_data.get("subexpr_cache"), dict):
            imported_data["subexpr_cache"] = {}

        # Replace the global cache
        _persistent_cache = imported_data

        # Save to disk
        save_cache_to_disk()

        if logger:
            logger.debug(f"Replaced cache from {file_path}")
        return True
    except (OSError, json.JSONDecodeError) as e:
        if logger:
            logger.warning(f"Failed to replace cache: {e}")
        return False
