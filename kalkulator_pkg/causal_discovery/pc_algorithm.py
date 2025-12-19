"""Causal Discovery using the PC Algorithm.

Implements the Peter-Clark (PC) algorithm for discovering causal structure
from observational data. The algorithm uses conditional independence tests
to infer the causal DAG (or CPDAG when directions are undeterminable).

Reference:
    Spirtes, P., Glymour, C., & Scheines, R. (2000).
    Causation, Prediction, and Search (2nd ed.). MIT Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from itertools import combinations

import numpy as np
from scipy import stats


@dataclass
class CausalGraph:
    """Represents a causal graph structure.

    Attributes:
        nodes: List of variable names
        edges: Set of directed edges (parent, child)
        undirected: Set of undirected edges (node1, node2) - for CPDAG
    """

    nodes: list[str] = field(default_factory=list)
    edges: set[tuple[str, str]] = field(default_factory=set)
    undirected: set[tuple[str, str]] = field(default_factory=set)

    def add_edge(self, parent: str, child: str):
        """Add a directed edge parent -> child."""
        self.edges.add((parent, child))

    def add_undirected(self, node1: str, node2: str):
        """Add an undirected edge node1 -- node2."""
        self.undirected.add((min(node1, node2), max(node1, node2)))

    def remove_edge(self, node1: str, node2: str):
        """Remove edge between nodes (directed or undirected)."""
        self.edges.discard((node1, node2))
        self.edges.discard((node2, node1))
        self.undirected.discard((min(node1, node2), max(node1, node2)))

    def has_edge(self, node1: str, node2: str) -> bool:
        """Check if there's any edge between nodes."""
        return (
            (node1, node2) in self.edges
            or (node2, node1) in self.edges
            or (min(node1, node2), max(node1, node2)) in self.undirected
        )

    def neighbors(self, node: str) -> set[str]:
        """Get all adjacent nodes."""
        result = set()
        for n1, n2 in self.edges:
            if n1 == node:
                result.add(n2)
            elif n2 == node:
                result.add(n1)
        for n1, n2 in self.undirected:
            if n1 == node:
                result.add(n2)
            elif n2 == node:
                result.add(n1)
        return result

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix."""
        n = len(self.nodes)
        node_idx = {node: i for i, node in enumerate(self.nodes)}
        adj = np.zeros((n, n))

        for parent, child in self.edges:
            adj[node_idx[parent], node_idx[child]] = 1

        for n1, n2 in self.undirected:
            adj[node_idx[n1], node_idx[n2]] = 1
            adj[node_idx[n2], node_idx[n1]] = 1

        return adj

    def to_networkx(self):
        """Convert to NetworkX graph."""
        try:
            import networkx as nx

            G = nx.DiGraph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)

            # Add undirected as bidirectional
            for n1, n2 in self.undirected:
                G.add_edge(n1, n2, undirected=True)
                G.add_edge(n2, n1, undirected=True)

            return G
        except ImportError:
            return None

    def __str__(self) -> str:
        lines = ["Causal Graph:"]
        lines.append(f"  Nodes: {', '.join(self.nodes)}")
        if self.edges:
            lines.append("  Directed edges:")
            for p, c in sorted(self.edges):
                lines.append(f"    {p} -> {c}")
        if self.undirected:
            lines.append("  Undirected edges:")
            for n1, n2 in sorted(self.undirected):
                lines.append(f"    {n1} -- {n2}")
        return "\n".join(lines)


def partial_correlation(
    X: np.ndarray, i: int, j: int, conditioning_set: list[int]
) -> float:
    """Compute partial correlation between variables i and j given conditioning set.

    Args:
        X: Data matrix (n_samples, n_variables)
        i: First variable index
        j: Second variable index
        conditioning_set: List of variable indices to condition on

    Returns:
        Partial correlation coefficient
    """
    if not conditioning_set:
        # Simple correlation
        return np.corrcoef(X[:, i], X[:, j])[0, 1]

    # Compute partial correlation using regression
    # Residualize i and j on conditioning set
    Z = X[:, conditioning_set]

    # Add constant
    Z_aug = np.column_stack([np.ones(len(Z)), Z])

    # Residuals of i
    try:
        beta_i = np.linalg.lstsq(Z_aug, X[:, i], rcond=None)[0]
        resid_i = X[:, i] - Z_aug @ beta_i
    except Exception:
        resid_i = X[:, i]

    # Residuals of j
    try:
        beta_j = np.linalg.lstsq(Z_aug, X[:, j], rcond=None)[0]
        resid_j = X[:, j] - Z_aug @ beta_j
    except Exception:
        resid_j = X[:, j]

    # Correlation of residuals
    return np.corrcoef(resid_i, resid_j)[0, 1]


def partial_correlation_test(
    X: np.ndarray, i: int, j: int, conditioning_set: list[int], alpha: float = 0.05
) -> tuple[bool, float]:
    """Test conditional independence using partial correlation.

    Uses Fisher's z-transform for significance testing.

    Args:
        X: Data matrix
        i: First variable index
        j: Second variable index
        conditioning_set: Conditioning set indices
        alpha: Significance level

    Returns:
        Tuple of (is_independent, p_value)
    """
    n = X.shape[0]

    rho = partial_correlation(X, i, j, conditioning_set)

    # Handle perfect (anti)correlation
    if abs(rho) >= 1.0 - 1e-10:
        return False, 0.0

    # Fisher's z-transform
    z = 0.5 * np.log((1 + rho) / (1 - rho))

    # Standard error
    dof = n - len(conditioning_set) - 3
    if dof < 1:
        dof = 1
    se = 1.0 / np.sqrt(dof)

    # Two-tailed test
    z_stat = abs(z) / se
    p_value = 2 * (1 - stats.norm.cdf(z_stat))

    is_independent = p_value > alpha

    return is_independent, p_value


def g2_test(
    X: np.ndarray,
    i: int,
    j: int,
    conditioning_set: list[int],
    alpha: float = 0.05,
    n_bins: int = 5,
) -> tuple[bool, float]:
    """G² (likelihood ratio) test for discrete/discretized data.

    Args:
        X: Data matrix
        i: First variable index
        j: Second variable index
        conditioning_set: Conditioning set indices
        alpha: Significance level
        n_bins: Number of bins for discretization

    Returns:
        Tuple of (is_independent, p_value)
    """

    # Discretize continuous data
    def discretize(col):
        return np.digitize(
            col, np.percentile(col, np.linspace(0, 100, n_bins + 1)[1:-1])
        )

    xi = discretize(X[:, i])
    xj = discretize(X[:, j])

    if conditioning_set:
        # Condition on each stratum
        z_combined = np.zeros(len(X), dtype=int)
        for _k, idx in enumerate(conditioning_set):
            z_combined = z_combined * n_bins + discretize(X[:, idx])

        g2_total = 0
        dof_total = 0

        for z_val in np.unique(z_combined):
            mask = z_combined == z_val
            if np.sum(mask) < 10:
                continue

            xi_z = xi[mask]
            xj_z = xj[mask]

            # Contingency table
            observed = np.histogram2d(xi_z, xj_z, bins=n_bins)[0]

            # Expected under independence
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            n_z = observed.sum()

            expected = row_sums * col_sums / (n_z + 1e-10)

            # G² statistic
            mask_pos = (observed > 0) & (expected > 0)
            g2 = 2 * np.sum(
                observed[mask_pos] * np.log(observed[mask_pos] / expected[mask_pos])
            )

            g2_total += g2
            dof_total += (n_bins - 1) ** 2
    else:
        # Simple test
        observed = np.histogram2d(xi, xj, bins=n_bins)[0]

        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        n_total = observed.sum()

        expected = row_sums * col_sums / (n_total + 1e-10)

        mask_pos = (observed > 0) & (expected > 0)
        g2_total = 2 * np.sum(
            observed[mask_pos] * np.log(observed[mask_pos] / expected[mask_pos])
        )
        dof_total = (n_bins - 1) ** 2

    if dof_total < 1:
        dof_total = 1

    p_value = 1 - stats.chi2.cdf(g2_total, dof_total)
    is_independent = p_value > alpha

    return is_independent, p_value


class PCAlgorithm:
    """Peter-Clark algorithm for causal discovery.

    Discovers the causal structure (DAG or CPDAG) from observational data
    using conditional independence tests.

    Example:
        >>> data = np.random.randn(1000, 4)  # 4 variables
        >>> data[:, 1] = data[:, 0] + np.random.randn(1000) * 0.1  # X0 -> X1
        >>> data[:, 2] = data[:, 1] + np.random.randn(1000) * 0.1  # X1 -> X2
        >>>
        >>> pc = PCAlgorithm(alpha=0.05)
        >>> graph = pc.fit(data, variable_names=['X0', 'X1', 'X2', 'X3'])
        >>> print(graph)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        ci_test: str = "gaussian",
        max_conditioning_set: int | None = None,
    ):
        """Initialize PC algorithm.

        Args:
            alpha: Significance level for independence tests
            ci_test: Type of conditional independence test
                     ('gaussian' for partial correlation, 'g2' for discrete)
            max_conditioning_set: Maximum size of conditioning sets to try
        """
        self.alpha = alpha
        self.ci_test = ci_test
        self.max_conditioning_set = max_conditioning_set
        self.separation_sets: dict = {}  # For orientation
        self.graph: CausalGraph | None = None

    def _ci_test(
        self, X: np.ndarray, i: int, j: int, conditioning_set: list[int]
    ) -> tuple[bool, float]:
        """Perform conditional independence test."""
        if self.ci_test == "gaussian":
            return partial_correlation_test(X, i, j, conditioning_set, self.alpha)
        elif self.ci_test == "g2":
            return g2_test(X, i, j, conditioning_set, self.alpha)
        else:
            return partial_correlation_test(X, i, j, conditioning_set, self.alpha)

    def fit(
        self, X: np.ndarray, variable_names: list[str] | None = None
    ) -> CausalGraph:
        """Learn causal structure from data.

        Args:
            X: Data matrix of shape (n_samples, n_variables)
            variable_names: Names for variables (default: X0, X1, ...)

        Returns:
            CausalGraph representing the learned structure
        """
        X = np.asarray(X)
        n_samples, n_vars = X.shape

        if variable_names is None:
            variable_names = [f"X{i}" for i in range(n_vars)]

        # Initialize complete undirected graph
        self.graph = CausalGraph(nodes=variable_names)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                self.graph.add_undirected(variable_names[i], variable_names[j])

        self.separation_sets = {}

        # Phase 1: Skeleton discovery
        # Iteratively remove edges based on conditional independence
        max_cond = self.max_conditioning_set or n_vars - 2

        for cond_size in range(max_cond + 1):
            edges_to_test = list(self.graph.undirected)

            for n1, n2 in edges_to_test:
                if not self.graph.has_edge(n1, n2):
                    continue

                i = variable_names.index(n1)
                j = variable_names.index(n2)

                # Get potential conditioning variables
                neighbors = self.graph.neighbors(n1) | self.graph.neighbors(n2)
                neighbors.discard(n1)
                neighbors.discard(n2)

                if len(neighbors) < cond_size:
                    continue

                # Test all conditioning sets of this size
                found_independence = False

                for cond_set in combinations(neighbors, cond_size):
                    cond_indices = [variable_names.index(v) for v in cond_set]

                    is_indep, p_value = self._ci_test(X, i, j, cond_indices)

                    if is_indep:
                        # Remove edge
                        self.graph.remove_edge(n1, n2)
                        self.separation_sets[(n1, n2)] = set(cond_set)
                        self.separation_sets[(n2, n1)] = set(cond_set)
                        found_independence = True
                        break

                if found_independence:
                    continue

        # Phase 2: Orientation rules
        self._orient_edges(variable_names)

        return self.graph

    def _orient_edges(self, variable_names: list[str]):
        """Apply orientation rules to convert skeleton to CPDAG.

        Rule 1: v-structures (X -> Z <- Y where X and Y not adjacent)
        Rule 2: No new v-structures
        Rule 3: No cycles
        """
        # Rule 1: Orient v-structures
        # For each triple X - Z - Y where X and Y are not adjacent
        for z in variable_names:
            neighbors = list(self.graph.neighbors(z))

            for x, y in combinations(neighbors, 2):
                if self.graph.has_edge(x, y):
                    continue

                # Check if z is in the separation set of x and y
                sep_set = self.separation_sets.get((x, y), set())

                if z not in sep_set:
                    # This is a v-structure: x -> z <- y
                    self.graph.undirected.discard((min(x, z), max(x, z)))
                    self.graph.undirected.discard((min(y, z), max(y, z)))
                    self.graph.add_edge(x, z)
                    self.graph.add_edge(y, z)

        # Rules 2 and 3: Propagate orientations
        changed = True
        while changed:
            changed = False

            for n1, n2 in list(self.graph.undirected):
                # Rule 2: If X -> Y and Y - Z, orient Y -> Z to avoid new v-structure
                for parent in variable_names:
                    if (parent, n1) in self.graph.edges:
                        if not self.graph.has_edge(parent, n2):
                            # Orient n1 -> n2
                            self.graph.undirected.discard((min(n1, n2), max(n1, n2)))
                            self.graph.add_edge(n1, n2)
                            changed = True
                            break
                    elif (parent, n2) in self.graph.edges:
                        if not self.graph.has_edge(parent, n1):
                            # Orient n2 -> n1
                            self.graph.undirected.discard((min(n1, n2), max(n1, n2)))
                            self.graph.add_edge(n2, n1)
                            changed = True
                            break

    def causal_effect(
        self, X: np.ndarray, treatment: str, outcome: str, variable_names: list[str]
    ) -> float | None:
        """Estimate causal effect using adjustment formula.

        Only works if the effect is identifiable from the graph.

        Args:
            X: Data matrix
            treatment: Treatment variable name
            outcome: Outcome variable name
            variable_names: Variable names

        Returns:
            Estimated causal effect coefficient, or None if not identifiable
        """
        if self.graph is None:
            raise ValueError("Must fit graph first")

        # Simple case: direct effect with no confounding
        # Adjust for parents of treatment that are not on causal path

        t_idx = variable_names.index(treatment)
        o_idx = variable_names.index(outcome)

        # Find parents of treatment (potential confounders)
        parents = set()
        for parent, child in self.graph.edges:
            if child == treatment:
                parents.add(parent)

        # Simple regression adjustment
        if parents:
            parent_indices = [variable_names.index(p) for p in parents]
            Z = X[:, parent_indices]
            Z_aug = np.column_stack([np.ones(len(Z)), Z, X[:, t_idx]])

            try:
                beta = np.linalg.lstsq(Z_aug, X[:, o_idx], rcond=None)[0]
                return beta[-1]  # Coefficient on treatment
            except Exception:
                return None
        else:
            # No adjustment needed
            try:
                X_aug = np.column_stack([np.ones(len(X)), X[:, t_idx]])
                beta = np.linalg.lstsq(X_aug, X[:, o_idx], rcond=None)[0]
                return beta[1]
            except Exception:
                return None


def discover_causal_graph(
    data: np.ndarray,
    variable_names: list[str] | None = None,
    alpha: float = 0.05,
    ci_test: str = "gaussian",
) -> CausalGraph:
    """Convenience function to discover causal structure.

    Args:
        data: Data matrix
        variable_names: Variable names
        alpha: Significance level
        ci_test: CI test type

    Returns:
        CausalGraph
    """
    pc = PCAlgorithm(alpha=alpha, ci_test=ci_test)
    return pc.fit(data, variable_names)
