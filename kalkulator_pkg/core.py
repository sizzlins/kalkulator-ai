"""Core module for Kalkulator AI.

This module defines the central `Context` object used to pass state explicitly
through the application pipeline, satisfying the "Remove Global State" audit requirement.
"""
from __future__ import annotations

import threading
from typing import Any, Optional
from dataclasses import dataclass, field

from .registry import FunctionRegistry
# from .config import Config # Config is currently module-level, keep it as is for now or move to Context?
# Audit requirement only mentioned FunctionRegistry singleton. Let's start with that.

@dataclass
class Context:
    """Application Context holding shared state.
    
    This object should be passed explicitly to functions requiring access to
    global state (registries, configuration, etc.), eliminating hidden
    dependencies on singletons.
    """
    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    
    # Future extensibility for other global states (e.g. specialized config, session ID)
    session_id: str = "default"
    
    def __post_init__(self):
        """Initialize any complex state if needed."""
        pass
