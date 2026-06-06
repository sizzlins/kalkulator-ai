"""NSGA-II Selection for Multi-Objective Symbolic Regression.

Implements Non-dominated Sorting Genetic Algorithm II (NSGA-II) for
proper multi-objective optimization of accuracy vs complexity.

References:
    Deb, K., et al. "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
    IEEE Transactions on Evolutionary Computation, 2002.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .expression_tree import ExpressionTree


@dataclass
class RankedIndividual:
    """Individual with NSGA-II ranking info."""
    tree: Any  # ExpressionTree
    fitness: float  # MSE (objective 1)
    complexity: int  # Node count (objective 2)
    rank: int = 0  # Pareto front rank (0 = best)
    crowding_distance: float = 0.0
    semantic_hash: str = ""  # v3.3: Functional equivalence signature


# SEMANTIC_PROBE_POINTS: Fixed random points for functional hashing (v3.3 Audit)
# Used to detect functionally equivalent trees (e.g., x+x and 2*x)
_SEMANTIC_SEED = 42
_SEMANTIC_N_PROBES = 10
_SEMANTIC_PROBES: np.ndarray | None = None


def compute_semantic_hash(tree, n_vars: int = 3) -> str:
    """Compute semantic hash by evaluating tree on fixed probe points.
    
    v3.3 Audit Remediation: Detects functional equivalence.
    Trees like 'x+x' and '2*x' will produce the same hash.
    
    Args:
        tree: ExpressionTree to hash
        n_vars: Number of variables in data (for probe dimensions)
        
    Returns:
        String hash of rounded evaluation results
    """
    global _SEMANTIC_PROBES
    
    # Lazy-initialize probe points (fixed seed for reproducibility)
    if _SEMANTIC_PROBES is None or _SEMANTIC_PROBES.shape[1] != n_vars:
        rng = np.random.default_rng(_SEMANTIC_SEED)
        _SEMANTIC_PROBES = rng.uniform(-10, 10, size=(_SEMANTIC_N_PROBES, n_vars))
    
    try:
        # Evaluate tree on probe points
        results = tree.evaluate_fast(_SEMANTIC_PROBES)
        if np.isscalar(results):
            results = np.full(_SEMANTIC_N_PROBES, results)
        
        # Round to 4 decimals to handle floating point differences
        rounded = np.round(results, 4)
        
        # Handle NaN/Inf by replacing with sentinel values
        rounded = np.where(np.isnan(rounded), 1e99, rounded)
        rounded = np.where(np.isinf(rounded), 1e98 * np.sign(rounded), rounded)
        
        return str(tuple(rounded))
    except Exception:
        # If evaluation fails, use structural hash as fallback
        return str(id(tree))


def fast_non_dominated_sort(population: list[RankedIndividual]) -> list[list[int]]:
    """Fast non-dominated sorting (NSGA-II Algorithm).
    
    Divides population into fronts where:
    - Front 0: Non-dominated by anyone
    - Front 1: Dominated only by Front 0
    - Front n: Dominated only by Fronts 0..n-1
    
    Time complexity: O(MN^2) where M=objectives, N=population
    
    Returns:
        List of fronts, each containing indices of individuals
    """
    n = len(population)
    if n == 0:
        return []
    
    # domination_count[i] = number of individuals that dominate i
    domination_count = [0] * n
    # dominated_set[i] = set of individuals that i dominates
    dominated_set = [[] for _ in range(n)]
    
    fronts = [[]]
    
    for i in range(n):
        p = population[i]
        for j in range(n):
            if i == j:
                continue
            q = population[j]
            
            # Check if p dominates q (lower is better for both objectives)
            p_dominates_q = (
                p.fitness <= q.fitness and p.complexity <= q.complexity and
                (p.fitness < q.fitness or p.complexity < q.complexity)
            )
            q_dominates_p = (
                q.fitness <= p.fitness and q.complexity <= p.complexity and
                (q.fitness < p.fitness or q.complexity < p.complexity)
            )
            
            if p_dominates_q:
                dominated_set[i].append(j)
            elif q_dominates_p:
                domination_count[i] += 1
        
        # If i is not dominated by anyone, it's in front 0
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    # Build subsequent fronts
    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = front_idx + 1
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)
    
    # Remove empty last front
    if not fronts[-1]:
        fronts.pop()
    
    return fronts


def calculate_crowding_distance(population: list[RankedIndividual], front: list[int]) -> None:
    """Calculate crowding distance for individuals in a front.
    
    Crowding distance measures how close an individual is to its neighbors.
    Higher distance = more isolated = more diverse = should be preserved.
    
    Args:
        population: Full population list
        front: Indices of individuals in this front
    """
    n = len(front)
    if n == 0:
        return
    
    # Reset distances
    for i in front:
        population[i].crowding_distance = 0.0
    
    if n <= 2:
        # Boundary points get infinite distance
        for i in front:
            population[i].crowding_distance = float('inf')
        return
    
    # For each objective (fitness, complexity)
    for get_objective in [lambda x: x.fitness, lambda x: x.complexity]:
        # Sort front by this objective
        sorted_front = sorted(front, key=lambda i: get_objective(population[i]))
        
        # Boundary points
        population[sorted_front[0]].crowding_distance = float('inf')
        population[sorted_front[-1]].crowding_distance = float('inf')
        
        # Get range for normalization
        obj_min = get_objective(population[sorted_front[0]])
        obj_max = get_objective(population[sorted_front[-1]])
        obj_range = obj_max - obj_min
        
        if obj_range < 1e-10 or not np.isfinite(obj_range):
            continue  # All same value or infinite range
        
        # Interior points
        for k in range(1, n - 1):
            prev_obj = get_objective(population[sorted_front[k - 1]])
            next_obj = get_objective(population[sorted_front[k + 1]])
            population[sorted_front[k]].crowding_distance += (next_obj - prev_obj) / obj_range


def nsga2_select(population: list, n_select: int) -> list:
    """NSGA-II selection operator with Diversity Enforcement.
    
    Selects individuals based on:
    1. Pareto rank (lower is better)
    2. Crowding distance (higher is better)
    
    Features explicit deduplication:
    - Identical individuals (same fitness/complexity) group together
    - Only one representative per group is ranked/distanced
    - Duplicates are assigned Crowding Distance = 0.0 (penalizing redundancy)
    
    Args:
        population: List of ExpressionTree objects
        n_select: Number of individuals to select
        
    Returns:
        List of selected individuals
    """
    if len(population) <= n_select:
        return population[:]
    
    # 1. Wrap and Group by SEMANTIC Signature (v3.3 Audit Remediation)
    # Semantic hash detects functional equivalence (e.g., x+x vs 2*x)
    # Signature = (fitness, complexity, semantic_hash) -> List[RankedIndividual]
    grouped_population = {}
    all_ranked = []
    
    # Determine n_vars from first individual
    n_vars = 1
    if population and hasattr(population[0], 'variable_names'):
        n_vars = len(population[0].variable_names)
    
    for ind in population:
        fit = getattr(ind, 'fitness', float('inf'))
        comp = ind.complexity() if hasattr(ind, 'complexity') else ind.root.count_nodes()
        
        # Compute semantic hash for functional equivalence (v3.3)
        sem_hash = compute_semantic_hash(ind, n_vars)
        
        ri = RankedIndividual(tree=ind, fitness=fit, complexity=comp, semantic_hash=sem_hash)
        all_ranked.append(ri)
        
        # Group by semantic signature (NOT just fitness/complexity)
        sig = (fit, comp, sem_hash)
        if sig not in grouped_population:
            grouped_population[sig] = []
        grouped_population[sig].append(ri)
    
    # 2. Extract Representatives (Unique Population)
    unique_ranked = [group[0] for group in grouped_population.values()]
    
    # 3. Perform Non-Dominated Sorting on Uniques
    fronts = fast_non_dominated_sort(unique_ranked)
    
    # 4. Calculate Crowding Distance for Uniques
    for front in fronts:
        calculate_crowding_distance(unique_ranked, front)
    
    # 5. Propagate Ranks and Handle Duplicates
    # unique_ranked now has correct .rank and .crowding_distance
    # We propagate rank to duplicates, but set their CD to 0.0 to prefer diversity
    final_population = []
    
    for unique in unique_ranked:
        sig = (unique.fitness, unique.complexity, unique.semantic_hash)
        duplicates = grouped_population[sig]
        
        # Add the representative (keeps calculated CD)
        final_population.append(unique)
        
        # Add the duplicates (Rank = same, CD = 0.0)
        for dup in duplicates[1:]:
            dup.rank = unique.rank
            dup.crowding_distance = 0.0  # Penalize redundancy
            final_population.append(dup)
            
    # 6. Sort by (Rank ASC, Crowding Distance DESC)
    # This naturally selects:
    # - Better fronts first
    # - Within front: Diverse individuals first
    # - Within same point: Representative first, then duplicates (if needed)
    final_population.sort(key=lambda x: (x.rank, -x.crowding_distance))
    
    # 7. Select Top N
    selected = [ri.tree for ri in final_population[:n_select]]
    return selected


    return selected


def assign_nsga2_ranks(population: list[ExpressionTree]) -> None:
    """Calculate and assign NSGA-II rank and crowding_distance to trees.
    
    This function modifies the trees in-place, adding `_nsga2_rank` 
    and `_nsga2_cd` attributes. This avoids re-sorting during tournament.
    
    Args:
        population: List of ExpressionTree objects
    """
    if not population:
        return

    # Wrap and rank
    ranked = [
        RankedIndividual(
            tree=ind,
            fitness=getattr(ind, 'fitness', float('inf')),
            complexity=ind.complexity() if hasattr(ind, 'complexity') else ind.root.count_nodes()
        )
        for ind in population
    ]
    
    fronts = fast_non_dominated_sort(ranked)
    for front in fronts:
        calculate_crowding_distance(ranked, front)
        
    # Assign back to trees
    for r in ranked:
        r.tree._nsga2_rank = r.rank
        r.tree._nsga2_cd = r.crowding_distance


def tournament_select_ranked(population: list[ExpressionTree], tournament_size: int = 2) -> ExpressionTree:
    """Select one individual using pre-calculated NSGA-II ranks with age bias.
    
    Requires `assign_nsga2_ranks(population)` to be called first.
    
    Args:
        population: List of ExpressionTree objects (with _nsga2_rank/cd attributes)
        tournament_size: Size of tournament
        
    Returns:
        Selected ExpressionTree
    """
    import random
    
    # Pick random candidates
    # Optimization: Use indices to avoid creating list of objects
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    
    # AFPO-inspired: 15% of the time, select the youngest individual regardless of rank
    # This prevents the MSE gravity well from instantly killing promising complex mutations!
    if random.random() < 0.15:
        def get_age(idx):
            return getattr(population[idx], 'age', float('inf'))
        youngest_idx = min(indices, key=get_age)
        return population[youngest_idx]
    
    # helper to get attributes safely
    def get_rank_cd(idx):
        t = population[idx]
        # Default to worst if not assigned (shouldn't happen if initialized properly)
        return getattr(t, '_nsga2_rank', float('inf')), -getattr(t, '_nsga2_cd', 0.0)

    # Winner has lower rank, or same rank with higher crowding distance
    winner_idx = min(indices, key=get_rank_cd)
    return population[winner_idx]
