
import unittest
import sympy as sp
from kalkulator_pkg.parser import _validate_expression_tree
from kalkulator_pkg.types import ValidationError

class TestListValidation(unittest.TestCase):
    def test_valid_list(self):
        try:
            # Should not raise
            _validate_expression_tree([1, 2, 3])
            _validate_expression_tree([[1, 2], [3, 4]])
            print("PASS: Valid lists accepted.")
        except ValidationError as e:
            self.fail(f"Valid list rejected: {e}")

    def test_valid_tuple(self):
        try:
             # Should not raise
             _validate_expression_tree((1, 2, 3))
             print("PASS: Valid tuples accepted.")
        except ValidationError as e:
             self.fail(f"Valid tuple rejected: {e}")

    def test_forbidden_type_in_list(self):
        # We need to mock an object that isn't allowed
        class BadType:
            pass
        
        with self.assertRaises(ValidationError) as cm:
            _validate_expression_tree([1, BadType()])
        print(f"PASS: Rejected list with bad type: {cm.exception}")
        
        self.assertIn("FORBIDDEN_TYPE", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
