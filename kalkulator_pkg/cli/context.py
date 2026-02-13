from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

from ..core import Context

# Global context singleton for worker access
_current_context: Optional["ReplContext"] = None


def get_current_context() -> Optional["ReplContext"]:
    """Get the current global REPL context.
    
    Returns:
        The current ReplContext if set, None otherwise.
    """
    return _current_context


def set_current_context(ctx: Optional["ReplContext"]) -> None:
    """Set the current global REPL context.
    
    Args:
        ctx: The ReplContext to set as current, or None to clear.
    """
    global _current_context
    _current_context = ctx


@dataclass
class ReplContext(Context):
    """Holds the state of the interactive REPL session.
    Inherits from core.Context to include Application State (FunctionRegistry).
    """

    timing_enabled: bool = False
    cache_hits_enabled: bool = False
    logging_enabled: bool = False
    debug_mode: bool = False
    current_req_id: Optional[str] = None
    cache_hits_tracking: List[Tuple[str, str]] = field(default_factory=list)
    banned_operators: set[str] = field(default_factory=set)

