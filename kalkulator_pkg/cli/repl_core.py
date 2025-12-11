
import sys
import logging
import re
import argparse
import time
import json
from typing import Any, Optional



from ..config import VAR_NAME_RE
from .context import ReplContext
from ..parser import split_top_level_commas, ALLOWED_SYMPY_NAMES
from ..function_manager import (
    define_variable,
    define_function,
    parse_function_definition,
    find_function_from_data
)
from ..worker import evaluate_safely
from ..solver import solve_single_equation, solve_system
from ..utils.formatting import format_solution, print_result_pretty
from ..utils.numeric import solve_modulo_system_if_applicable

logger = logging.getLogger(__name__)

class REPL:
    """
    Modular REPL implementation replacing the monolithic repl_loop.
    Adheres to Engineering Standards: Small Units, Linear Logic, Encapsulation.
    """
    def __init__(self, context: Optional[ReplContext] = None):
        self.ctx = context if context else ReplContext()
        self.running = True
        self.chained_context: dict[str, str] = {}
        self.variables: dict[str, str] = {} # Global variable cache for substitution
        self.results_buffer: list[str] = []
        self._setup_readline()
        
    def _setup_readline(self):
        try:
            import readline
        except ImportError:
            pass

    def _get_allowed_functions(self, text: str) -> frozenset[str] | None:
        """
        Detect undefined names used as functions to allow them in parsing.
        Useful for function finding (e.g., 'f(1)=2').
        """
        # Find pattern 'name('
        # exclude specific keywords if needed, but ALLOWED_SYMPY_NAMES handles most
        candidates = set()
        for match in re.finditer(r"\b([a-zA-Z_]\w*)\s*\(", text):
            name = match.group(1)
            if name not in ALLOWED_SYMPY_NAMES and name not in self.variables:
                candidates.add(name)
        
        return frozenset(candidates) if candidates else None

    def start(self):
        """Main loop entry point."""
        from ..config import VERSION
        
        # We can just print a simple welcome here or define it
        print(f"kalkulator-ai v{VERSION} — type 'help' for commands, 'quit' to exit.")
        
        while self.running:
            self.loop_once()
            
    def loop_once(self):
        """Single iteration of the read-eval-print loop."""
        try:
            prompt = ">>> " if not self.ctx.debug_mode else "DEBUG>>> "
            try:
                raw = input(prompt)
            except EOFError:
                self.running = False
                return
                
            self.process_input(raw)
        except KeyboardInterrupt:
            self.handle_interrupt()
        except Exception as e:
            logger.exception("Unexpected error in REPL loop")
            print(f"Error: {e}")

    def handle_interrupt(self):
        if self.ctx.current_req_id:
             # Logic to cancel request would go here if we tracked requests completely
             # For now, just print
             print("\n[Interrupted]")
        else:
            print("\n[Press Ctrl+C again to exit]")

    def process_input(self, text: str):
        """Dispatch input to specific handlers."""
        text = text.strip()
        if not text:
            return

        # 1. Check for Commands (quit, help, etc.) via Registry
        # We pass self.variables so commands can modify it (like clear)
        # or use it (like export substituting values?? No export uses internal function manager)
        # But 'handle_command' needs access to variables for 'solve' warning logic.
        from .repl_commands import handle_command
        if handle_command(text, self.ctx, self.variables):
            return
            
        # Legacy check for quit logic if not handled in handle_command (it wasn't)
        if text.lower() in ("quit", "exit"):
            self.running = False
            return
        if text.lower() in ("help", "?"):
            from .app import print_help_text
            print_help_text()
            return

        # 1b. Check for "Function Finding" pattern (e.g. "f(1)=1, find f(x)")
        # If detected, treat the WHOLE string as a 'find' command to avoid splitting.
        if "find" in text.lower() and "=" in text:
             # Dispatch to repl_commands which handles parsing the data points
             from .repl_commands import handle_find_command_raw
             if handle_find_command_raw(text, self.ctx):
                 return

        # 2. Check for Function Definitions
        
        # 3. Check for Chains/Systems via comma splitting
        if "," in text:
            parts = split_top_level_commas(text)
            if len(parts) > 1:
                self._handle_multi_part_input(parts, text)
                return
                
        # 4. Fallback: Single Expression/Assignment/Definition handling
        self._handle_single_part(text) # force_solve handled in handle_command or via solve_single_equation?
        # Wait, if handle_command handles "solve", then _handle_single_part doesn't need force_solve logic?
        # Correct. "solve x^2=4" is handled by handle_command now.
        return

    def _try_handle_command(self, text: str) -> bool:
        """Deprecated: Logic moved to commands.py and process_input."""
        return False

    def _substitute_variables(self, text: str) -> str:
        """Substitute global variables into text."""
        # Sort variables by length descending to specific ambiguous prefix issues
        sorted_vars = sorted(self.variables.keys(), key=len, reverse=True)
        for var in sorted_vars:
            val = self.variables[var]
            if var in text:
                pattern = r'\b' + re.escape(var) + r'\b'
                text = re.sub(pattern, f"({val})", text)
        return text

    def _handle_multi_part_input(self, parts: list[str], raw_text: str):
        """
        Decide between System Solving and Chained Execution.
        Rule 1: Simple Control Flow.
        """
        # Detection Logic:
        # We want to chain if ALL parts are explicit Assignments OR Definitions.
        # Otherwise, we treat it as a System of Equations (modulo, linear, etc.)
        
        is_sequential_list = True
        
        for p in parts:
            p = p.strip()
            if "=" not in p:
                # If no equals, it's likely an expression or bad input.
                # If it's an expression like "x+y", we usually EVALUATE it if it's single.
                # If it's "x=1, x+y", we might chain.
                # But for now, let's assume chain requires assignments/defs?
                # Actually, "x=1, x+5" is valid chain: x=1 then print 6.
                pass 
            else:
                lhs = p.split("=", 1)[0].strip()
                # Check variable assignment
                if VAR_NAME_RE.match(lhs):
                    continue
                # Check function definition
                if parse_function_definition(p):
                    continue
                
                # If neither, it is a complex equation (e.g. x+y=10)
                is_sequential_list = False
                break
        
        # Check for same-variable system (e.g. x=1 mod 2, x=2 mod 3)
        same_variable_system = False
        if is_sequential_list:
             # Gather assigned vars (only vars, not functions)
             assigned_vars = []
             for p in parts:
                 if "=" in p:
                     lhs = p.split("=", 1)[0].strip()
                     if VAR_NAME_RE.match(lhs):
                         assigned_vars.append(lhs)
             
             if len(set(assigned_vars)) == 1 and len(assigned_vars) > 1:
                 same_variable_system = True

        if same_variable_system:
             # Delegate to modulo solver
             var = parts[0].split("=", 1)[0].strip()
             solved, _ = solve_modulo_system_if_applicable(parts, var, "human")
             if solved: 
                 return

        # Auto-detect Function Finding intent (e.g. "f(1)=2, f(2)=3")
        # If users provide data points for an undefined function but forget "find f(x)", we should help them.
        from .repl_commands import handle_find_command_raw
        
        # Quick heuristic: if we have multiple equations, and they look like f(const)=val
        candidate_func_names = set()
        is_data_pattern = True
        
        for p in parts:
            p = p.strip()
            if "=" not in p: 
                is_data_pattern = False; break
            
            lhs = p.split("=", 1)[0].strip()
            # Must look like func call
            match = re.match(r"^([a-zA-Z_]\w*)\s*\(", lhs)
            if not match:
                is_data_pattern = False; break
            
            name = match.group(1)
            # If name is already defined as variable, it's not a function finding task (probably)
            # If defined as function, we might be refining it? But usually finding implies unknown.
            if name in self.variables:
                is_data_pattern = False; break
                
            candidate_func_names.add(name)
            
        if is_data_pattern and len(candidate_func_names) == 1:
            target_func = list(candidate_func_names)[0]
            # Construct synthetic command
            # We assume 1D f(x) for simplicity in auto-detection, or let handle_find_command_raw infer
            # handle_find_command_raw expects "text" containing data and "find ...".
            # We append ", find target_func(x)" to safe-guard detection.
            # handle_find_command_raw parses commas itself.
            
            # We pass original text + find command
            enhanced_text = raw_text + f", find {target_func}(x)"
            # Print helpful message
            print(f"Auto-detecting function finding for '{target_func}'...")
            
            if handle_find_command_raw(enhanced_text, self.ctx):
                return

        if is_sequential_list:
            self._execute_chain(parts)
        else:
            # System solver
            subbed_text = self._substitute_variables(raw_text)
            allowed = self._get_allowed_functions(raw_text)
            res = solve_system(subbed_text, None, allowed_functions=allowed)
            print_result_pretty(res)

    def _execute_chain(self, parts: list[str]):
        """Execute a chain of commands/assignments/definitions with persistence."""
        self.chained_context = {} # Refresh local chain context
        self.results_buffer = []
        
        for part in parts:
            part = part.strip()
            
            # Check for definition FIRST (before global substitution mangles params)
            func_def_candidate = parse_function_definition(part)
            if func_def_candidate:
                name, params, body = func_def_candidate
                # Substitute ONLY in body
                body_subbed = self._substitute_variables(body)
                body_subbed = self._substitute_chain_context(body_subbed)
                
                try:
                    define_function(name, params, body_subbed)
                    self.results_buffer.append(f"Function '{name}' defined")
                    continue
                except Exception as e:
                    self.results_buffer.append(f"Error defining '{name}': {e}")
                    continue

            # Standard Assignment/Expression handling
            # Substitute Global
            part_subbed = self._substitute_variables(part)
            # Substitute Local
            part_subbed = self._substitute_chain_context(part_subbed)
            
            # Execute
            if "=" in part:
                 self._handle_chain_assignment(part, part_subbed)
            else:
                 self._handle_chain_expression(part, part_subbed)
                 
        if self.results_buffer:
            print(", ".join(self.results_buffer))

    def _substitute_chain_context(self, text: str) -> str:
        # Similar to variables but for chained_context
        # Sort keys by length
        sorted_keys = sorted(self.chained_context.keys(), key=len, reverse=True)
        for var in sorted_keys:
            val = self.chained_context[var]
            if var in text:
                pattern = r'\b' + re.escape(var) + r'\b'
                text = re.sub(pattern, f"({val})", text)
        return text

    def _handle_chain_assignment(self, raw_part: str, subbed_part: str):
        # We need the original variable name for definition
        if "=" not in raw_part:
            # Fallback if logic mismatch
            self._handle_chain_expression(raw_part, subbed_part)
            return

        var_name = raw_part.split("=", 1)[0].strip()
        # The RHS is what needs substitution
        rhs_subbed = subbed_part.split("=", 1)[1].strip()
        
        res = evaluate_safely(rhs_subbed)
        if res.get("ok"):
            val_str = res.get("result")
            # 1. Update Chain Context
            self.chained_context[var_name] = val_str
            # 2. Update Global Variables
            self.variables[var_name] = val_str
            # 3. Persist Globally (Backing store)
            try:
                define_variable(var_name, val_str)
            except Exception:
                pass
            
            # 4. Buffer Output
            approx = res.get("approx")
            item = f"{var_name} = {format_solution(val_str)}"
            # (Omitted complex formatting logic for brevity on first pass, adhering to clarity)
            self.results_buffer.append(item)
        else:
            # Error in assignment
            self.results_buffer.append(f"Error in '{raw_part}': {res.get('error')}")

    def _handle_chain_expression(self, raw_part: str, subbed_part: str):
        # Treat as expression OR legacy command (like find z)
        # If persistence works, solve_single_equation("z") fails if no =.
        if "=" in subbed_part:
            allowed = self._get_allowed_functions(raw_part)
            res = solve_single_equation(subbed_part, None, allowed_functions=allowed)
        else:
            # Just evaluate e.g. "z" or "x+y"
            res = evaluate_safely(subbed_part)
            if res.get("ok"):
                 val = format_solution(res.get("result", "")) # result is string
                 self.results_buffer.append(f"{raw_part} = {val}")
                 return
            # If evaluate failed, pass through (error printing logic below)
        
        if res.get("ok"):
            # This path is for solve_single_equation success
            # solve_single_equation returns 'exact' list
            vals = res.get("exact", [])
            val_str = ", ".join(vals)
            self.results_buffer.append(f"{raw_part} = {val_str}")
        else:
             # Buffer error? Or define error item?
             self.results_buffer.append(f"{raw_part} = Error: {res.get('error')}")

    def _handle_single_part(self, text: str, force_solve: bool = False):
        """Handle a single statement (assignment, definition, expression)."""
        # 0. Global Substitution (if not defining new function/var on LHS)
        # If text is "y = x + 5", we want "y = (10) + 5"
        # BUT we must NOT substitute lhs "y" if it's being defined!
        # So we can't blindly substitute 'text'.
        
        # 1. Check Function Definition
        if "=" in text and not text.startswith("solve") and not force_solve: # Heuristic
             # Need checking if it's f(x)=...
             func_def = parse_function_definition(text)
             # Substitutions in body? Yes.
             if func_def:
                 name, params, body = func_def
                 # We probably don't want to substitute in body for function definition? 
                 # Or do we? "f(t) = x*t" -> "f(t) = 10*t"? 
                 # Usually YES in simple REPLs (capture value).
                 # Simulating capture by substitution.
                 body_subbed = self._substitute_variables(body)
                 try:
                     define_function(name, params, body_subbed)
                     print(f"Function '{name}' defined.")
                     return
                 except Exception as e:
                     print(f"Error defining function: {e}")
                     return

        if "=" in text and not text.startswith("solve") and not force_solve:
             # Check for simple variable assignment: var = expr
             # We already checked for function def f(x)=..., but let's be robust
             lhs, rhs = text.split("=", 1)
             lhs = lhs.strip()
             rhs = rhs.strip()
             
             if VAR_NAME_RE.match(lhs):
                 # It is an assignment!
                 # Substitute RHS
                 rhs_subbed = self._substitute_variables(rhs)
                 res = evaluate_safely(rhs_subbed)
                 if res.get("ok"):
                     val_str = res.get("result")
                     try:
                         define_variable(lhs, val_str)
                         self.variables[lhs] = val_str # Update global cache
                         print(f"{lhs} = {format_solution(val_str)}")
                         return
                     except Exception as e:
                         print(f"Error defining variable: {e}")
                         return
                 else:
                      print(f"Error evaluating assignment: {res.get('error')}")
                      return

        if any(op in text for op in ("<", ">", "<=", ">=")):
             text_subbed = self._substitute_variables(text)
             allowed = self._get_allowed_functions(text)
             res = solve_single_equation(text_subbed, None, allowed_functions=allowed) # Or inequality solver
             print_result_pretty(res)
             return

        # 3. Default Solve/Eval
        text_subbed = self._substitute_variables(text)
        
        if "=" in text_subbed or force_solve:
             allowed = self._get_allowed_functions(text)
             res = solve_single_equation(text_subbed, None, allowed_functions=allowed)
             print_result_pretty(res)
        else:
             # Evaluate expression
             allowed = self._get_allowed_functions(text)
             res = evaluate_safely(text_subbed, allowed_functions=allowed)
             # Wrap in pretty print format, ensuring 'result' key matches formatting.py expectations
             if res.get("ok"):
                 print_result_pretty({
                     "ok": True,
                     "type": "evaluation",
                     "result": res.get("result"), # KEY FIX
                     "exact": [res.get("result")],
                     "approx": [res.get("approx")]
                 })
             else:
                 print_result_pretty(res)

