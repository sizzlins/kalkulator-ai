import os
from pathlib import Path

# Define files to include in the packet
files_to_include = [
    'kalkulator_pkg/symbolic_regression/genetic_engine.py',
    'kalkulator_pkg/symbolic_regression/genetic_config.py',
    'kalkulator_pkg/symbolic_regression/strategies.py',
    'kalkulator_pkg/symbolic_regression/forensic_analysis.py',
    'kalkulator_pkg/symbolic_regression/expression_tree.py',
    'kalkulator_pkg/symbolic_regression/operators.py',
    'kalkulator_pkg/parser.py',
    'kalkulator_pkg/tokenizer.py',
    'kalkulator_pkg/worker.py',
    'kalkulator_pkg/function_manager.py',
    'kalkulator_pkg/config.py',
    'kalkulator_pkg/cli/repl_core.py',
    'kalkulator_pkg/benchmarks/feynman_equations.py',
]

output = ['# GEMINI REVIEW PACKET - Updated 2026-01-16']
output.append('# This packet contains critical files for security and architecture review.')
output.append('')
output.append('# ================================================================================')
output.append('# SECURITY SUMMARY')
output.append('# ================================================================================')
output.append('# The following security hardening has been implemented:')
output.append('# 1. parser.py: Replaced sympy.parse_expr (eval-based) with safe_sympy_parse (AST-based)')
output.append('#    - SafeSymPyVisitor class uses Python ast module to build SymPy expressions')
output.append('#    - Blocks: __import__, eval, attribute access (x.__class__), etc.')
output.append('# 2. function_manager.py: Uses safe_sympy_parse for registry loading')
output.append('# 3. feynman_equations.py: Added AST validation before benchmark lambda compilation')
output.append('# 4. tokenizer.py: FORBIDDEN_NAMES blocklist for __import__, eval, exec, etc.')
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
