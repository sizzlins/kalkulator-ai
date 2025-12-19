"""Type definitions and result dataclasses for consistent API responses."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a mathematical expression."""

    ok: bool
    result: str | None = None
    approx: str | None = None
    free_symbols: list[str] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict: dict[str, Any] = {"ok": self.ok}
        if self.result is not None:
            result_dict["result"] = self.result
        if self.approx is not None:
            result_dict["approx"] = self.approx
        if self.free_symbols is not None:
            result_dict["free_symbols"] = self.free_symbols
        if self.error is not None:
            result_dict["error"] = self.error
        return result_dict

    def __repr__(self) -> str:
        """Return string representation of the result."""
        if not self.ok:
            return f"EvalResult(ok=False, error={self.error!r})"
        parts = [f"ok={self.ok}"]
        if self.result is not None:
            parts.append(f"result={self.result!r}")
        if self.approx is not None:
            parts.append(f"approx={self.approx!r}")
        if self.free_symbols is not None:
            parts.append(f"free_symbols={self.free_symbols!r}")
        return f"EvalResult({', '.join(parts)})"


@dataclass
class SolveResult:
    """Result of solving an equation."""

    ok: bool
    result_type: str  # "equation", "pell", "identity_or_contradiction", "multi_isolate", "system"
    error: str | None = None
    # For equation type
    exact: list[str] | None = None
    approx: list[str | None] | None = None
    # For pell type
    solution: str | None = None
    # For multi_isolate type
    solutions: dict[str, list[str]] | None = None
    # For system type
    system_solutions: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict: dict[str, Any] = {"ok": self.ok, "type": self.result_type}
        if self.error is not None:
            result_dict["error"] = self.error
        if self.exact is not None:
            result_dict["exact"] = self.exact
        if self.approx is not None:
            result_dict["approx"] = self.approx
        if self.solution is not None:
            result_dict["solution"] = self.solution
        if self.solutions is not None:
            result_dict["solutions"] = self.solutions
        if self.system_solutions is not None:
            result_dict["solutions"] = self.system_solutions
        return result_dict

    def __repr__(self) -> str:
        """Return string representation of the result."""
        if not self.ok:
            return f"SolveResult(ok=False, result_type={self.result_type!r}, error={self.error!r})"
        parts = [f"ok={self.ok}", f"result_type={self.result_type!r}"]
        if self.exact is not None:
            parts.append(f"exact={self.exact!r}")
        if self.approx is not None:
            parts.append(f"approx={self.approx!r}")
        if self.solution is not None:
            parts.append(f"solution={self.solution!r}")
        if self.solutions is not None:
            parts.append(f"solutions={self.solutions!r}")
        if self.system_solutions is not None:
            parts.append(f"system_solutions={self.system_solutions!r}")
        return f"SolveResult({', '.join(parts)})"


@dataclass
class InequalityResult:
    """Result of solving an inequality."""

    ok: bool

    result_type: str = field(default="inequality")
    error: str | None = None
    solutions: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict: dict[str, Any] = {"ok": self.ok, "type": self.result_type}
        if self.error is not None:
            result_dict["error"] = self.error
        if self.solutions is not None:
            result_dict["solutions"] = self.solutions
        return result_dict

    def __repr__(self) -> str:
        """Return string representation of the result."""
        if not self.ok:
            return f"InequalityResult(ok=False, error={self.error!r})"
        parts = [f"ok={self.ok}", f"result_type={self.result_type!r}"]
        if self.solutions is not None:
            parts.append(f"solutions={self.solutions!r}")
        return f"InequalityResult({', '.join(parts)})"


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ParseError(Exception):
    """Raised when parsing fails."""

    def __init__(self, message: str, code: str = "PARSE_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class SolverError(Exception):
    """Raised when solving fails."""

    def __init__(
        self, message: str, code: str = "SOLVER_ERROR", transient: bool = False
    ):
        self.message = message
        self.code = code
        self.transient = transient
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
