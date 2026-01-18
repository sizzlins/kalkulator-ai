from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

from ..core import Context

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
