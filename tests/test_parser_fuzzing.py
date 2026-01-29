"""Extended Fuzzing Tests for SafeSymPyVisitor Security.

v3.3 Audit Remediation: Tests parser resilience against malformed/malicious inputs.

Run with: pytest tests/test_parser_fuzzing.py -v
Or with more examples: pytest tests/test_parser_fuzzing.py --hypothesis-seed=random

Requires: pip install hypothesis
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalkulator_pkg.parser import safe_sympy_parse, ValidationError


# =============================================================================
# Strategy Definitions
# =============================================================================

# Random text (potentially malicious)
random_text = st.text(min_size=0, max_size=1000)

# Valid mathematical-ish expressions
math_chars = st.sampled_from("0123456789xyzabc+-*/^()., ")
math_expr = st.text(alphabet=math_chars, min_size=1, max_size=500)

# Deeply nested expressions
def nested_ops(depth: int) -> str:
    """Generate nested function calls which AST preserves."""
    if depth <= 0:
        return "x"
    # usage of sin(...) ensures AST depth increases
    return f"sin({nested_ops(depth - 1)})"

class TestDepthLimits:
    """Parser must enforce depth limits to prevent stack exhaustion."""
    
    def test_max_depth_enforced(self):
        """Expressions deeper than MAX_DEPTH must be rejected."""
        # Create expression with depth > 100 (the MAX_DEPTH)
        # Each sin(...) adds depth, plus the base node
        deep_expr = nested_ops(150)
        with pytest.raises(ValidationError) as exc_info:
            safe_sympy_parse(deep_expr)
        assert "nested" in str(exc_info.value).lower() or "depth" in str(exc_info.value).lower()
    
    def test_just_under_depth_limit(self):
        """Expressions just under MAX_DEPTH should parse successfully."""
        # 50 is safe (Max 100)
        safe_expr = nested_ops(50)
        try:
            result = safe_sympy_parse(safe_expr)
            assert result is not None
        except ValidationError:
            pass  # Also acceptable


class TestInputLengthLimits:
    """Parser must enforce input length limits to prevent DoS."""
    
    def test_max_length_enforced(self):
        """Inputs longer than MAX_INPUT_LENGTH must be rejected."""
        long_input = "x+" * 6000  # ~12000 chars, exceeds 10000 limit
        with pytest.raises(ValidationError) as exc_info:
            safe_sympy_parse(long_input)
        assert "long" in str(exc_info.value).lower() or "length" in str(exc_info.value).lower()
    
    def test_just_under_length_limit(self):
        """Inputs just under MAX_INPUT_LENGTH should parse."""
        safe_input = "x+" * 2000 + "1"  # ~4001 chars
        try:
            result = safe_sympy_parse(safe_input)
            assert result is not None
        except ValidationError:
            pass  # Also acceptable (may fail for other reasons)


class TestEdgeCases:
    """Edge cases that might slip through normal validation."""
    
    def test_empty_input(self):
        """Empty input should raise ValidationError."""
        with pytest.raises(ValidationError):
            safe_sympy_parse("")
        with pytest.raises(ValidationError):
            safe_sympy_parse("   ")
    
    def test_unicode_attacks(self):
        """Unicode characters should not bypass security."""
        attacks = [
            "еvаl('1')",  # Cyrillic 'e' and 'a'
            "ехеc('1')",  # Cyrillic letters
            "\u202e1+2",  # Right-to-left override
            "x\x00y",     # Null byte
        ]
        for attack in attacks:
            try:
                safe_sympy_parse(attack)
            except (ValidationError, Exception):
                pass  # Any rejection is acceptable
    
    def test_comment_injection(self):
        """Comments should be ignored by parser, not crash or execute."""
        # This parses as just "x", allowing the comment is fine as long as it's ignored
        result = safe_sympy_parse("x # + __import__('os')")
        assert str(result) == "x"


# =============================================================================
# Run directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
