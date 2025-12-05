"""Optional plotting functionality for single-variable functions."""

from __future__ import annotations

import sympy as sp

try:
    # Set non-GUI backend before importing pyplot to avoid Tkinter issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend (no Tkinter required)
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
    HAS_GUI_BACKEND = False  # We're using Agg backend
except ImportError:
    HAS_MATPLOTLIB = False
    HAS_GUI_BACKEND = False
except Exception:
    # If setting backend fails, try to continue anyway
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
        HAS_GUI_BACKEND = True
    except ImportError:
        HAS_MATPLOTLIB = False
        HAS_GUI_BACKEND = False

from .parser import parse_preprocessed
from .types import EvalResult
from .worker import evaluate_safely


def _open_file_in_viewer(file_path: str) -> bool:
    """Open a file in the system's default application (cross-platform).
    
    Args:
        file_path: Path to the file to open
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import os
        import sys
        import subprocess
        
        if sys.platform == "win32":
            # Windows
            os.startfile(file_path)
            return True
        elif sys.platform == "darwin":
            # macOS
            subprocess.run(["open", file_path], check=True)
            return True
        else:
            # Linux and other Unix-like systems
            subprocess.run(["xdg-open", file_path], check=True)
            return True
    except Exception:
        # Silently fail - opening is a convenience feature
        return False


def plot_function(
    expression: str,
    variable: str = "x",
    x_min: float = -10,
    x_max: float = 10,
    points: int = 100,
    ascii: bool = False,
) -> EvalResult:
    """Plot a single-variable function.
    
    This function creates a visual plot of a mathematical expression using matplotlib.
    If matplotlib is not available or GUI backend fails, the plot is automatically
    saved to a temporary file and opened in the default image viewer.
    
    Features:
    - Automatic file saving if GUI display is unavailable
    - Enhanced styling with grid, axes, and legend
    - Support for mathematical expressions in range parameters
    - Cross-platform file opening (Windows, macOS, Linux)
    
    Args:
        expression: Function expression to plot (e.g., "x^2", "sin(x)", "exp(-x^2)")
        variable: Variable name to plot against (default: "x")
        x_min: Minimum x value for plot range (default: -10)
        x_max: Maximum x value for plot range (default: 10)
        points: Number of points to sample for plotting (default: 100)
        ascii: If True, return ASCII plot text; If False, use matplotlib (default: False)

    Returns:
        EvalResult with:
        - ok=True: Plot was created successfully
        - result: Path to saved plot file or "Plot displayed" message
        - ok=False: Plotting failed
        - error: Error message describing the failure
        
    Examples:
        >>> from kalkulator_pkg.plotting import plot_function
        >>> result = plot_function("x^2", x_min=-5, x_max=5)
        >>> print(result.result)  # "Plot saved and opened: /tmp/plot.png"
        
        >>> result = plot_function("sin(x)", x_min=-pi, x_max=pi, ascii=True)
        >>> print(result.result)  # ASCII plot text
    """
    if not HAS_MATPLOTLIB and not ascii:
        return EvalResult(
            ok=False, error="matplotlib not installed. Use ascii=True for ASCII plot."
        )

    try:
        eval_result = evaluate_safely(expression)
        if not eval_result.get("ok"):
            return EvalResult(ok=False, error=eval_result.get("error"))

        expr = parse_preprocessed(eval_result["result"])
        var_sym = sp.symbols(variable)

        # Create numeric evaluation function
        f = sp.lambdify(var_sym, expr, "numpy")

        if ascii:
            # ASCII plot
            import numpy as np

            x_vals = np.linspace(x_min, x_max, points)
            y_vals = f(x_vals)

            # Simple ASCII representation
            # Constants for ASCII plot dimensions
            rows = 20  # Height of ASCII plot in characters
            cols = 60  # Width of ASCII plot in characters
            plot_chars = [[" " for _ in range(cols)] for _ in range(rows)]

            # Find y range
            valid_y = [y for y in y_vals if y == y and -1e10 < y < 1e10]
            if not valid_y:
                return EvalResult(
                    ok=False, error="Cannot plot: function values out of range"
                )

            y_min, y_max = min(valid_y), max(valid_y)
            y_range = y_max - y_min if y_max != y_min else 1

            # Plot points
            for x, y in zip(x_vals, y_vals):
                if y != y or not (-1e10 < y < 1e10):
                    continue
                col = int((x - x_min) / (x_max - x_min) * (cols - 1))
                row = int((y_max - y) / y_range * (rows - 1))
                col = max(0, min(cols - 1, col))
                row = max(0, min(rows - 1, row))
                plot_chars[row][col] = "*"

            # Add axes
            x_axis_row = (
                int((0 - y_min) / y_range * (rows - 1)) if y_min <= 0 <= y_max else -1
            )
            y_axis_col = (
                int((0 - x_min) / (x_max - x_min) * (cols - 1))
                if x_min <= 0 <= x_max
                else -1
            )

            ascii_plot = []
            for r in range(rows):
                line = []
                for c in range(cols):
                    if r == x_axis_row and c == y_axis_col:
                        line.append("+")
                    elif r == x_axis_row:
                        line.append("-")
                    elif c == y_axis_col:
                        line.append("|")
                    else:
                        line.append(plot_chars[rows - 1 - r][c])
                ascii_plot.append("".join(line))

            plot_text = "\n".join(ascii_plot)
            return EvalResult(ok=True, result=f"ASCII plot:\n{plot_text}")

        else:
            # Matplotlib plot with enhanced styling
            import numpy as np

            x_vals = np.linspace(x_min, x_max, points)
            try:
                y_vals = f(x_vals)
            except (ValueError, TypeError, ZeroDivisionError):
                # Handle cases where function might fail at certain points
                # Try to evaluate point by point
                y_vals = []
                for x in x_vals:
                    try:
                        y = float(sp.N(expr.subs(var_sym, x)))
                        y_vals.append(y)
                    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                        y_vals.append(np.nan)
                y_vals = np.array(y_vals)

            # Create figure with better styling
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, linewidth=2, color='#2E86AB', label=f'f({variable}) = {expression}')
            ax.set_xlabel(variable, fontsize=12, fontweight='bold')
            ax.set_ylabel(f'f({variable})', fontsize=12, fontweight='bold')
            ax.set_title(f'Plot of {expression}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Improve layout
            plt.tight_layout()
            
            # Try to show plot, but if GUI backend fails, save to temp file instead
            try:
                if HAS_GUI_BACKEND:
                    plt.show()
                    return EvalResult(ok=True, result="Plot displayed")
                else:
                    # Using non-GUI backend, save to temp file and inform user
                    import tempfile
                    import os
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    plt.savefig(temp_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Try to open the file automatically
                    opened = _open_file_in_viewer(temp_path)
                    if opened:
                        return EvalResult(ok=True, result=f"Plot saved and opened: {temp_path}")
                    else:
                        return EvalResult(ok=True, result=f"Plot saved to: {temp_path}\n(Note: GUI backend not available. File opened in default viewer.)")
            except Exception as e:
                # If showing fails, try to save instead
                try:
                    import tempfile
                    import os
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    plt.savefig(temp_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Try to open the file automatically
                    opened = _open_file_in_viewer(temp_path)
                    if opened:
                        return EvalResult(ok=True, result=f"Plot saved and opened: {temp_path}\n(GUI display failed, using file output)")
                    else:
                        return EvalResult(ok=True, result=f"Plot saved to: {temp_path}\n(GUI display failed: {str(e)})")
                except Exception as save_error:
                    plt.close(fig)
                    return EvalResult(ok=False, error=f"Failed to display or save plot: {str(e)}")

    except (ValueError, TypeError, AttributeError) as e:
        return EvalResult(ok=False, error=f"Plotting error: {e}")
    except ImportError:
        return EvalResult(ok=False, error="NumPy required for plotting")
    except (ZeroDivisionError, OverflowError) as e:
        return EvalResult(ok=False, error=f"Numerical error in plotting: {e}")
    except Exception as e:
        try:
            from .logging_config import get_logger

            logger = get_logger("plotting")
            logger.error(f"Unexpected plotting error: {e}", exc_info=True)
        except ImportError:
            pass
        return EvalResult(ok=False, error="Plotting failed unexpectedly")
