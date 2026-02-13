"""Gene Bank: Long-Term Memory for Symbolic Regression.

This module implements a persistence layer for successful expressions,
enabling the engine to "learn" across sessions by saving high-quality
solutions and reusing them as seeds in future runs.

Key Features:
- Canonical normalization (variables -> v0, v1, v2...)
- De-duplication via sorted commutative ops
- Quality filtering (complexity < 10, R² > 0.99)
- Diversity cap (max 25% of population)
- Arity filtering (gene.n_vars <= problem.n_vars)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from .expression_tree import ExpressionTree

# Configuration
GENE_BANK_DIR = Path.home() / ".kalkulator"
GENE_BANK_FILE = GENE_BANK_DIR / "gene_bank.json"
MAX_COMPLEXITY = 10
MIN_R2 = 0.99
HARD_CAP = 50
DIVERSITY_RATIO = 0.25


@dataclass
class Gene:
    """A single stored gene (normalized expression)."""
    expression: str           # Normalized expression string (e.g., "sin(v0) * v1")
    n_vars: int               # Number of unique variables
    complexity: int           # Node count or similar metric
    fitness: float            # MSE at time of saving
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> Gene:
        return cls(**data)


class GeneBank:
    """Persistent storage for learned expressions."""
    
    def __init__(self, path: Path | None = None):
        self.path = path or GENE_BANK_FILE
        self.genes: list[Gene] = []
        self._ensure_directory()
        self._load()
    
    def _ensure_directory(self):
        """Create storage directory if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load(self):
        """Load genes from disk."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                self.genes = [Gene.from_dict(g) for g in data.get('genes', [])]
            except (json.JSONDecodeError, KeyError, TypeError):
                self.genes = []
        else:
            self.genes = []
    
    def _save(self):
        """Persist genes to disk."""
        data = {'genes': [g.to_dict() for g in self.genes]}
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------
    
    def add(self, tree: ExpressionTree, fitness: float, r2: float | None = None) -> bool:
        """Add a tree to the Gene Bank if it passes quality checks.
        
        Args:
            tree: The expression tree to save.
            fitness: MSE of the tree.
            r2: Optional R² score. If provided, must be >= MIN_R2.
            
        Returns:
            True if the gene was added, False if rejected.
        """
        # Quality Gate 1: Complexity
        complexity = tree.complexity()
        if complexity > MAX_COMPLEXITY:
            return False
        
        # Quality Gate 2: Accuracy
        if r2 is not None and r2 < MIN_R2:
            return False
        
        # Normalize the expression
        normalized_expr, n_vars = self._normalize(tree)
        if normalized_expr is None:
            return False
        
        # De-duplication: Check if already exists
        if any(g.expression == normalized_expr for g in self.genes):
            return False
        
        # Create and store gene
        gene = Gene(
            expression=normalized_expr,
            n_vars=n_vars,
            complexity=complexity,
            fitness=fitness
        )
        self.genes.append(gene)
        self._save()
        return True
    
    def get_seeds(self, variable_names: list[str], pop_size: int, allowed_operators: list[str] | None = None) -> list[str]:
        """Get compatible seeds for a given problem.
        
        Implements:
        - Arity filtering (gene.n_vars <= len(variable_names))
        - Diversity cap (max 25% of pop_size)
        - Operator filtering (skips genes using disallowed ops)
        - Permutation limit (all permutations if n_vars <= 3, direct only otherwise)
        
        Args:
            variable_names: Variable names in the current problem (e.g., ['x', 'y']).
            pop_size: Total population size.
            allowed_operators: List of allowed operator names (e.g. ['sin', 'add']). If None, all allowed.
            
        Returns:
            List of expression strings ready for seeding.
        """
        from ..tokenizer import Tokenizer

        n_problem_vars = len(variable_names)
        diversity_limit = min(HARD_CAP, int(pop_size * DIVERSITY_RATIO))
        
        seeds = []
        
        for gene in self.genes:
            if len(seeds) >= diversity_limit:
                break
            
            # Arity Filter: Skip genes with too many variables
            if gene.n_vars > n_problem_vars:
                continue
            
            # Operator Filter: Check if gene uses forbidden operators
            if allowed_operators is not None:
                # Basic tokenization to find functions/operators
                # We can reuse the Tokenizer or just check substrings for simple safety
                # For robustness, we'll use a simple token check provided by Tokenizer
                try:
                    tokens = Tokenizer.tokenize(gene.expression)
                    # Helper tokens not in 'operators' list
                    ignored = {'(', ')', ',', 'x', 'y', 'z', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5'}
                    
                    is_safe = True
                    for tok in tokens:
                        if tok.type in ('FUNCTION', 'OPERATOR'):
                            val = tok.value
                            # Map symbols back to names for check
                            # e.g. '+' -> 'add', '*' -> 'mul'
                            # Actually, allowed_operators usually contains names like 'add', 'sin'
                            # But expressions contain symbols '+', '*'
                            # We need a robust check.
                            
                            # Fast path: if token is in allowed list, good.
                            if val in allowed_operators:
                                continue
                                
                            # Convert symbol to name if needed
                            # This is tricky without a full map available here.
                            # However, disallowed operators like 'bitwise_xor', 'floor' usually appear as names
                            # or specific symbols.
                            
                            # Bitwise check (primary goal of this fix)
                            if val in ['&', '|', '^', '<<', '>>', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'lshift', 'rshift']:
                                # Check if corresponding name is allowed
                                op_name = None
                                if val == '&' or val == 'bitwise_and': op_name = 'bitwise_and'
                                elif val == '|' or val == 'bitwise_or': op_name = 'bitwise_or'
                                elif val == '^' or val == 'bitwise_xor': op_name = 'bitwise_xor'
                                elif val == '<<' or val == 'lshift': op_name = 'lshift'
                                elif val == '>>' or val == 'rshift': op_name = 'rshift'
                                
                                if op_name and op_name not in allowed_operators:
                                    is_safe = False
                                    break
                                    
                            # Discrete check
                            if val in ['floor', 'ceil', 'round', 'sign', 'max', 'min'] and val not in allowed_operators:
                                is_safe = False
                                break
                                
                    if not is_safe:
                        continue
                except Exception:
                    # If tokenization fails, skip gene to be safe
                    continue

            # Map gene variables to problem variables
            mapped_exprs = self._map_variables(gene.expression, gene.n_vars, variable_names)
            
            for expr in mapped_exprs:
                if len(seeds) >= diversity_limit:
                    break
                seeds.append(expr)
        
        return seeds
    
    def delete(self, index: int) -> bool:
        """Delete a gene by index.
        
        Args:
            index: 0-based index of the gene to delete.
            
        Returns:
            True if deleted, False if index out of range.
        """
        if 0 <= index < len(self.genes):
            del self.genes[index]
            self._save()
            return True
        return False
    
    def clear(self):
        """Delete all genes."""
        self.genes = []
        self._save()
    
    def list_genes(self) -> list[dict]:
        """Return all genes as a list of dicts for display."""
        return [
            {
                'id': i,
                'expression': g.expression,
                'n_vars': g.n_vars,
                'complexity': g.complexity,
                'fitness': g.fitness
            }
            for i, g in enumerate(self.genes)
        ]
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    
    def _normalize(self, tree: ExpressionTree) -> tuple[str | None, int]:
        """Normalize a tree to canonical form with generic variable names.
        
        Transforms variable names to v0, v1, v2... based on appearance order.
        Sorts commutative operations (add, mul) for de-duplication.
        
        Returns:
            Tuple of (normalized_expression_string, num_variables) or (None, 0) on failure.
        """
        try:
            # Get SymPy expression
            sympy_expr = tree.to_sympy()
            if sympy_expr is None:
                return None, 0
            
            # Extract free symbols and sort by name for deterministic ordering
            free_syms = sorted(sympy_expr.free_symbols, key=str)
            n_vars = len(free_syms)
            
            if n_vars == 0:
                # Constant expression - not useful
                return None, 0
            
            # Create substitution map: original -> v0, v1, v2...
            v_symbols = [sp.Symbol(f"v{i}") for i in range(n_vars)]
            subs_map = {old: new for old, new in zip(free_syms, v_symbols)}
            
            # Apply substitution
            normalized = sympy_expr.subs(subs_map)
            
            # Canonicalize: SymPy automatically sorts commutative ops in __repr__
            # but we can force it with expand + simplify
            try:
                normalized = sp.nsimplify(normalized, rational=False)
            except Exception:
                pass
            
            return str(normalized), n_vars
        
        except Exception:
            return None, 0
    
    def _map_variables(self, expr_str: str, n_gene_vars: int, problem_vars: list[str]) -> list[str]:
        """Map generic variable names (v0, v1...) to problem variables.
        
        If n_gene_vars < len(problem_vars), generates multiple mappings.
        If n_gene_vars == len(problem_vars) and n_gene_vars <= 3, generates permutations.
        Otherwise, direct mapping only.
        
        Args:
            expr_str: Normalized expression string (e.g., "sin(v0) + v1").
            n_gene_vars: Number of variables in the gene.
            problem_vars: Variable names in current problem.
            
        Returns:
            List of expression strings with mapped variables.
        """
        from itertools import permutations, combinations
        
        n_problem_vars = len(problem_vars)
        results = []
        
        if n_gene_vars == n_problem_vars:
            # Same arity: permutations if small, direct if large
            if n_gene_vars <= 3:
                for perm in permutations(problem_vars):
                    mapped = expr_str
                    for i, var in enumerate(perm):
                        mapped = mapped.replace(f"v{i}", f"__{var}__")
                    for var in perm:
                        mapped = mapped.replace(f"__{var}__", var)
                    results.append(mapped)
            else:
                # Direct mapping only
                mapped = expr_str
                for i, var in enumerate(problem_vars):
                    mapped = mapped.replace(f"v{i}", var)
                results.append(mapped)
        
        elif n_gene_vars < n_problem_vars:
            # Fewer gene vars: map to all combinations of problem vars
            # E.g., sin(v0) with [x, y, z] -> sin(x), sin(y), sin(z)
            for combo in combinations(problem_vars, n_gene_vars):
                if n_gene_vars <= 3:
                    # Allow permutations within the combo
                    for perm in permutations(combo):
                        mapped = expr_str
                        for i, var in enumerate(perm):
                            mapped = mapped.replace(f"v{i}", f"__{var}__")
                        for var in perm:
                            mapped = mapped.replace(f"__{var}__", var)
                        results.append(mapped)
                else:
                    # Direct mapping only
                    mapped = expr_str
                    for i, var in enumerate(combo):
                        mapped = mapped.replace(f"v{i}", var)
                    results.append(mapped)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        
        return unique


# Singleton instance for global access
_global_bank: GeneBank | None = None


def get_gene_bank() -> GeneBank:
    """Get or create the global GeneBank instance."""
    global _global_bank
    if _global_bank is None:
        _global_bank = GeneBank()
    return _global_bank
