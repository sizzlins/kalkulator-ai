
from typing import Any, List, Tuple, Dict, Optional
import numpy as np
import logging
import warnings

from .core import Context
from .utils.numeric import eval_to_float
from .parser import safe_sympy_parse
from .finder_strategies import UnivariateStrategy, MultivariateStrategy, HybridStrategy

logger = logging.getLogger(__name__)

class FinderDispatch:
    """
    Central dispatcher for finding functions from data.
    Implements the Strategy Pattern to select appropriate solvers.
    """
    
    def __init__(
        self,
        context: Context,
        data_points: List[Tuple[Any, Any]],
        param_names: Optional[List[str]] = None,
        skip_linear: bool = False,
        verbose: bool = False
    ):
        self.context = context
        self.raw_data = data_points
        self.param_names = param_names or ["x"]
        self.config = {
            "skip_linear": skip_linear,
            "verbose": verbose,
            "use_genetic": True,
            "banned_operators": getattr(context, "banned_operators", None),
        }
        
        # Partitioned Data
        self.numeric_data: List[Tuple[Any, Any]] = []
        self.symbolic_data: List[Tuple[Any, Any]] = []
        self.complex_data: List[Tuple[Any, Any]] = []
        
        self.partitioned = False

    def _is_complex_value(self, val: Any) -> bool:
        """Check if a value is complex (has non-negligible imaginary part)."""
        if isinstance(val, (complex, np.complexfloating)):
            return abs(val.imag) > 1e-10
        if isinstance(val, str):
            # Check for imaginary indicators in string
            if any(indicator in val for indicator in ["i", "I", "*I", "j"]):
                try:
                    # Normalize and parse
                    val_normalized = val.replace("i", "*I").replace("jj", "*I")
                    parsed = safe_sympy_parse(val_normalized)
                    if hasattr(parsed, "as_real_imag"):
                        _, imag = parsed.as_real_imag()
                        return abs(float(imag.evalf())) > 1e-10
                except Exception:
                    # Assume complex if parsing fails but indicators present
                    return True
        if hasattr(val, "as_real_imag"):
            try:
                _, imag = val.as_real_imag()
                return abs(float(imag.evalf())) > 1e-10
            except Exception:
                pass
        return False

    def prepare_data(self) -> None:
        """Partition data into Numeric, Symbolic, and Complex sets."""
        if self.partitioned:
            return

        for x_tuple, y_val in self.raw_data:
            is_symbolic = False
            is_complex = False
            parsed_x_tuple = []

            # Process Inputs
            x_inputs = x_tuple if isinstance(x_tuple, (list, tuple, np.ndarray)) else [x_tuple]
            for x_arg in x_inputs:
                if self._is_complex_value(x_arg):
                    is_complex = True
                    break
                try:
                    # Suppress ComplexWarning for types like np.complex128(10+0j)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        val_float = float(x_arg)
                    parsed_x_tuple.append(val_float)
                except (ValueError, TypeError):
                    try:
                        val_float = eval_to_float(x_arg)
                        parsed_x_tuple.append(val_float)
                    except ValueError:
                        is_symbolic = True
                        try:
                            # Use safe parser for symbolic expression
                            expr = safe_sympy_parse(str(x_arg))
                            parsed_x_tuple.append(expr)
                        except Exception:
                            parsed_x_tuple.append(x_arg)

            # Process Output
            if not is_complex and self._is_complex_value(y_val):
                is_complex = True

            if is_complex:
                self.complex_data.append((x_tuple, y_val))
                continue

            if not is_symbolic:
                try:
                    y_float = eval_to_float(y_val)
                except ValueError:
                    is_symbolic = True
                    # Keep original y_val if it can't be floated
                    y_float = y_val # This variable is reused logic from monolith, messy but ok

            # Append to appropriate list
            if is_symbolic:
                y_store = y_val if is_symbolic else y_float
                self.symbolic_data.append((parsed_x_tuple, y_store))
            else:
                self.numeric_data.append((parsed_x_tuple, y_float))

        self.partitioned = True
        
        # Log complex data warning
        if self.complex_data:
            print(f"Warning: {len(self.complex_data)} data point(s) with complex/imaginary values were skipped.")
            print("         Regression currently requires real-valued inputs and outputs.")

    def solve(self) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Main entry point. 
        Selects strategy based on data dimensionality and types.
        """
        self.prepare_data()

        if not self.numeric_data and not self.symbolic_data:
            return (False, None, None, "No valid real-valued data points provided.")

        # Strategy Selection
        # 1. 1D Function
        if len(self.param_names) == 1:
            strategy = UnivariateStrategy()
            return strategy.solve(
                self.context, 
                self.numeric_data, 
                self.symbolic_data, 
                self.param_names, 
                self.config
            )
        
        # 2. Multi-D Function
        else:
            # Check for Hybrid needs
            if self.symbolic_data:
                strategy = HybridStrategy()
            else:
                strategy = MultivariateStrategy()
                
            return strategy.solve(
                self.context, 
                self.numeric_data, 
                self.symbolic_data, 
                self.param_names, 
                self.config
            )
