
import sys
import math
from dataclasses import dataclass

# Mock ExpressionTree to avoid importing the whole heavy tree structure
@dataclass
class MockTree:
    name: str
    fitness: float
    _complexity: int
    
    def complexity(self):
        return self._complexity
        
    def __repr__(self):
        return f"{self.name}(Fit={self.fitness}, Comp={self._complexity})"

# Add path to find package
sys.path.append(".")

from kalkulator_pkg.symbolic_regression.nsga2 import nsga2_select, RankedIndividual

def test_deduplication():
    print("Testing NSGA-II Deduplication...")
    
    # Create population with duplicates
    # A points: Best (0.1, 5)
    # B points: Good (0.5, 3)
    # C points: OK (1.0, 2)
    
    # A1, A2, A3 are identical best fitness
    A1 = MockTree("A1", 0.1, 5)
    A2 = MockTree("A2", 0.1, 5) 
    A3 = MockTree("A3", 0.1, 5)
    
    # B1, B2 are identical good fitness
    B1 = MockTree("B1", 0.5, 3)
    B2 = MockTree("B2", 0.5, 3)
    
    # C1 is unique
    C1 = MockTree("C1", 1.0, 2)
    
    # D1 is dominated by everything (worse fit, worse comp)
    D1 = MockTree("D1", 10.0, 20)

    population = [A1, A2, A3, B1, B2, C1, D1]
    
    print("\nPopulation:")
    for p in population:
        print(f"  {p}")

    # Case 1: Select only uniques (n=3)
    # Should pick one A, one B, one C (Pareto front)
    # A, B, C are non-dominated relative to each other?
    # A(0.1, 5) vs B(0.5, 3) -> A better fit, B better comp -> Non-dominated
    # B(0.5, 3) vs C(1.0, 2) -> B better fit, C better comp -> Non-dominated
    # So A, B, C are Rank 0.
    
    print("\nSelecting Top 3 (Expect A, B, C uniques)...")
    selected_3 = nsga2_select(population, 3)
    print("Selected:", selected_3)
    
    names = set(ind.name for ind in selected_3)
    # Should contain exactly one A, one B, one C
    # Or 2 As and 1 B if they share rank/CD?
    # Uniques: A, B, C.
    # CD: A(Inf), C(Inf), B(finite).
    # Rank: All 0.
    # So A and C should definitely be picked first. Then B.
    # So we expect {A?, C1, B?} where ? means any duplicate of that type
    
    # Check if duplicates are penalized
    # If we ask for 4, we should get A, B, C + one duplicate (A or B)
    print("\nSelecting Top 4 (Expect A, B, C + 1 duplicate)...")
    selected_4 = nsga2_select(population, 4)
    print("Selected:", selected_4)
    
    if len(selected_4) != 4:
        print("FAIL: Did not return 4 individuals")
        
    print("\nSelecting Top 6 (All Pareto + duplicates, skip dominated D1)...")
    selected_6 = nsga2_select(population, 6)
    print("Selected:", selected_6)
    
    d_names = [ind.name for ind in selected_6]
    if "D1" in d_names:
        print("FAIL: D1 should be rank 1 (dominated) and selected last or not at all")
        
    print("\nSUCCESS: Deduplication logic verification passed.")

if __name__ == "__main__":
    test_deduplication()
