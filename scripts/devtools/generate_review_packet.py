
import os
from pathlib import Path

# Define files to include in the packet
files_to_include = [
    # Core Symbolic Regression
    'kalkulator_pkg/symbolic_regression/genetic_engine.py',  # EvolutionTrainer class added
    'kalkulator_pkg/symbolic_regression/genetic_config.py',  # IntegerBiasWeighting rename
    'kalkulator_pkg/symbolic_regression/strategies.py',
    'kalkulator_pkg/symbolic_regression/forensic_analysis.py',
    'kalkulator_pkg/symbolic_regression/expression_tree.py',
    'kalkulator_pkg/symbolic_regression/operators.py',
    'kalkulator_pkg/symbolic_regression/parallel.py',  # Resource tracker fix
    'kalkulator_pkg/symbolic_regression/nsga2.py',  # Deduplication logic
    'kalkulator_pkg/symbolic_regression/numba_evaluator.py',  # Stack overflow guard (MAX_STACK=64)
    'kalkulator_pkg/symbolic_regression/constant_anchors.py', # [NEW]
    'kalkulator_pkg/symbolic_regression/pareto_front.py', # [NEW]
    'kalkulator_pkg/symbolic_regression/ode_discovery.py', # [NEW]
    'kalkulator_pkg/symbolic_regression/numerical_diff.py', # [NEW]
    'kalkulator_pkg/symbolic_regression/population.py', # [NEW]
    'kalkulator_pkg/symbolic_regression/symbolic_reconstruction.py', # [NEW]
    # Core Utilities
    'kalkulator_pkg/core.py', # [NEW] Context definition
    'kalkulator_pkg/parser.py',
    'kalkulator_pkg/tokenizer.py',
    'kalkulator_pkg/worker.py',  # WindowsJobObject context manager
    'kalkulator_pkg/function_manager.py',  # FinderDispatch
    'kalkulator_pkg/registry.py', # [NEW] FunctionRegistry Class
    'kalkulator_pkg/regression_solver.py',
    'kalkulator_pkg/config.py',
    'kalkulator_pkg/utils/lll.py',  # LLL stability guards
    # CLI
    'kalkulator_pkg/cli/app.py', # [NEW] Entry point & Help text
    'kalkulator_pkg/cli/repl_core.py',
    'kalkulator_pkg/cli/repl_commands.py',  # Loading feedback
    'kalkulator_pkg/cli/context.py', # [NEW] ReplContext inheriting core.Context
    # Benchmarks
    'kalkulator_pkg/benchmarks/feynman_equations.py',
]

output = ['# GEMINI REVIEW PACKET - Updated 2026-01-20 (Comprehensive v4.5)']
output.append('# This packet contains DEFINITIVE fixes for v3.6 Audit Findings (v4.2).')
output.append('')
output.append('# ================================================================================')
output.append('# V4.2 REMEDIATION SUMMARY (FINAL)')
output.append('# ================================================================================')
output.append('# 1. Concurrency: Explicit resource_tracker.unregister (Fixed "Resource Tracker Trap").')
output.append('# 2. Registry: Pickle-safe (Excluded RLock).')
output.append('# 3. Math: Negative Gradient Boosting & Exact LLL (Removed max_iter & Integer Snapping).')
output.append('# 4. Security: Removed Parsing Regex & Hardened Visitor (Fixed "Parser Vulnerability").')
output.append('# 5. Architecture: Decoupled God Object & Removed Lazy Imports.')
output.append('# 6. Hygiene: Proper Logging & Robust Windows Handles.')
output.append('#')
output.append('# ARCHITECTURAL (Code Quality):')
output.append('#  5. genetic_engine.py: EvolutionTrainer class (decoupled God Object)')
output.append('#  6. core.py: Context class (Removed Global State)')
output.append('#  7. function_manager.py: Refactored for Context Passing')
output.append('#  8. registry.py: FunctionRegistry class (Thread-safe)')
output.append('#')
output.append('# SCIENTIFIC (Accuracy & Bias):')
output.append('#  9. genetic_config.py: use_integer_anchoring=False (opt-in heuristics)')
output.append('# 10. nsga2.py: Signature grouping deduplication')
output.append('# 11. lll.py: Maximal stability (Exact/Float fallback guards)')
output.append('#')
output.append('# UX:')
output.append('# 12. repl_commands.py: "Loading genetic evolution engine..." feedback')
output.append('')

# Generate project structure
output.append('# PROJECT STRUCTURE')
output.append('```')
for root, dirs, files in os.walk('kalkulator_pkg'):
    dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]
    level = root.replace('kalkulator_pkg', '').count(os.sep)
    indent = '    ' * level
    output.append(f'{indent}{os.path.basename(root)}/')
    subindent = '    ' * (level + 1)
    for f in sorted(files):
        if f.endswith('.py'):
            output.append(f'{subindent}{f}')
output.append('```')
output.append('')

# Add file contents
for filepath in files_to_include:
    if os.path.exists(filepath):
        output.append('')
        output.append(f'# FILE: {filepath}')
        output.append('=' * 80)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Truncate very large files
            if len(content) > 50000:
                content = content[:50000] + '\n\n# ... [TRUNCATED FOR BREVITY - File is ' + str(len(content)) + ' bytes] ...'
            output.append(content)
        output.append('=' * 80)
        output.append('')

# Write output
with open('GEMINI_REVIEW_PACKET.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print(f'Generated GEMINI_REVIEW_PACKET.txt with {len(files_to_include)} files')
