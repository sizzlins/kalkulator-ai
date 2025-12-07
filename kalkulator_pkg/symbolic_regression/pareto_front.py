"""Pareto front for multi-objective optimization.

Maintains a set of Pareto-optimal solutions trading off accuracy vs complexity.
This is key for symbolic regression where we want the simplest model that fits well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParetoSolution:
    """A solution on the Pareto front.
    
    Attributes:
        expression: String representation of the expression
        sympy_expr: SymPy expression object
        mse: Mean squared error (fitness)
        complexity: Number of nodes in expression tree
        tree: Reference to the original ExpressionTree
    """
    expression: str
    sympy_expr: Any  # sp.Expr
    mse: float
    complexity: int
    tree: Any = field(compare=False)  # ExpressionTree
    
    def dominates(self, other: ParetoSolution) -> bool:
        """Check if this solution dominates another.
        
        A solution dominates another if it's at least as good in all objectives
        and strictly better in at least one.
        """
        # Lower is better for both MSE and complexity
        at_least_as_good = (self.mse <= other.mse and self.complexity <= other.complexity)
        strictly_better = (self.mse < other.mse or self.complexity < other.complexity)
        return at_least_as_good and strictly_better
    
    def __lt__(self, other: ParetoSolution) -> bool:
        """Compare by MSE first, then complexity."""
        if self.mse != other.mse:
            return self.mse < other.mse
        return self.complexity < other.complexity


class ParetoFront:
    """Maintains a set of Pareto-optimal solutions.
    
    The Pareto front contains solutions where no solution is dominated by another.
    This allows trading off accuracy vs complexity based on user preference.
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize empty Pareto front.
        
        Args:
            max_size: Maximum number of solutions to keep
        """
        self.solutions: list[ParetoSolution] = []
        self.max_size = max_size
    
    def add(self, solution: ParetoSolution) -> bool:
        """Add a solution to the front if it's not dominated.
        
        Args:
            solution: Solution to potentially add
            
        Returns:
            True if solution was added (is Pareto-optimal)
        """
        # Check if new solution is dominated by any existing
        for existing in self.solutions:
            if existing.dominates(solution):
                return False
        
        # Remove solutions dominated by the new one
        self.solutions = [
            s for s in self.solutions
            if not solution.dominates(s)
        ]
        
        # Add new solution
        self.solutions.append(solution)
        
        # Trim if too large (keep most diverse)
        if len(self.solutions) > self.max_size:
            self._trim()
        
        return True
    
    def _trim(self):
        """Trim front to max_size, keeping diverse solutions."""
        if len(self.solutions) <= self.max_size:
            return
        
        # Sort by complexity and keep evenly spaced
        self.solutions.sort(key=lambda s: s.complexity)
        
        # Keep solutions at regular intervals
        indices = np.linspace(0, len(self.solutions) - 1, self.max_size, dtype=int)
        self.solutions = [self.solutions[i] for i in indices]
    
    def get_best(self, complexity_budget: int | None = None) -> ParetoSolution | None:
        """Get the best solution, optionally under a complexity budget.
        
        Args:
            complexity_budget: Maximum allowed complexity (None for no limit)
            
        Returns:
            Best solution meeting the criteria, or None if empty
        """
        if not self.solutions:
            return None
        
        candidates = self.solutions
        if complexity_budget is not None:
            candidates = [s for s in candidates if s.complexity <= complexity_budget]
        
        if not candidates:
            # No solutions under budget, return simplest
            return min(self.solutions, key=lambda s: s.complexity)
        
        # Return lowest MSE among candidates
        return min(candidates, key=lambda s: s.mse)
    
    def get_simplest(self) -> ParetoSolution | None:
        """Get the simplest solution on the front."""
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: s.complexity)
    
    def get_most_accurate(self) -> ParetoSolution | None:
        """Get the most accurate solution on the front."""
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: s.mse)
    
    def get_knee_point(self) -> ParetoSolution | None:
        """Get the 'knee' of the Pareto front.
        
        The knee is the solution that offers the best trade-off between
        accuracy and complexity (maximum distance from the line connecting
        the extremes).
        """
        if len(self.solutions) < 3:
            return self.get_best()
        
        # Sort by complexity
        sorted_sols = sorted(self.solutions, key=lambda s: s.complexity)
        
        # Get extreme points
        simplest = sorted_sols[0]
        most_accurate = min(sorted_sols, key=lambda s: s.mse)
        
        # Line from simplest to most accurate
        # Normalize both axes
        complexities = np.array([s.complexity for s in sorted_sols])
        mses = np.array([s.mse for s in sorted_sols])
        
        c_range = max(complexities.max() - complexities.min(), 1)
        m_range = max(mses.max() - mses.min(), 1e-10)
        
        c_norm = (complexities - complexities.min()) / c_range
        m_norm = (mses - mses.min()) / m_range
        
        # Points of the line
        p1 = np.array([c_norm[0], m_norm[0]])
        p2 = np.array([c_norm[-1], m_norm[-1]])
        
        # Find point with maximum distance to line
        max_dist = -1
        knee_idx = 0
        
        for i in range(len(sorted_sols)):
            p = np.array([c_norm[i], m_norm[i]])
            # Distance from point to line
            dist = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return sorted_sols[knee_idx]
    
    def to_list(self) -> list[dict]:
        """Convert front to list of dicts for serialization."""
        return [
            {
                'expression': s.expression,
                'mse': s.mse,
                'complexity': s.complexity,
            }
            for s in sorted(self.solutions, key=lambda s: s.complexity)
        ]
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for visualization."""
        try:
            import pandas as pd
            return pd.DataFrame(self.to_list())
        except ImportError:
            return self.to_list()
    
    def __len__(self) -> int:
        return len(self.solutions)
    
    def __iter__(self):
        return iter(sorted(self.solutions, key=lambda s: s.complexity))
    
    def __repr__(self) -> str:
        return f"ParetoFront({len(self.solutions)} solutions)"
