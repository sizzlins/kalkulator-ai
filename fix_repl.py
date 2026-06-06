import sys

with open('c:/Users/LOQ/PycharmProjects/kalkulator-ai/kalkulator_pkg/cli/repl_commands.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
skip_call = False
for i, line in enumerate(lines):
    if line.strip().startswith('def _handle_evolve('):
        new_lines.append('def _handle_evolve(text: str, ctx, variables=None):\n')
        new_lines.append('    from kalkulator_pkg.cli.handlers.evolution import handle_evolve\n')
        new_lines.append('    return handle_evolve(text, ctx, variables)\n')
        skip = True
        continue
        
    if skip and line.strip().startswith('def _handle_save_cache('):
        skip = False
        
    if not skip:
        if 'if " call " in raw_lower or " callr " in raw_lower:' in line:
            skip_call = True
            continue
            
        if skip_call and line.strip().startswith('# Check for Dynamic Shortcuts'):
            skip_call = False
            
        if not skip_call:
            new_lines.append(line)

with open('c:/Users/LOQ/PycharmProjects/kalkulator-ai/kalkulator_pkg/cli/repl_commands.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
