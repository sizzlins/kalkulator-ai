from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class ReplContext:
    """Holds the state of the interactive REPL session."""
    timing_enabled: bool = False
    cache_hits_enabled: bool = False
    logging_enabled: bool = False
    debug_mode: bool = False
    current_req_id: Optional[str] = None
    cache_hits_tracking: List[Tuple[str, str]] = field(default_factory=list)
