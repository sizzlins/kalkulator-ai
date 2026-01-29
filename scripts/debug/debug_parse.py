import re

POINT_PATTERN = re.compile(r'^([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*=\s*(.+)$')

parts = [
    'f(0)=12.2',  'f(0.05)=14.1', 
    'f(0.35)=26.4f(0)=12.2',  # This is the problem
    'f(0.05)=14.1'
]

all_points = []
for p in parts:
    p = p.strip()
    m = POINT_PATTERN.match(p)
    if m:
        name = m.group(1)
        args_str = m.group(2)
        val_str = m.group(3)
        args = [a.strip() for a in args_str.split(',')]
        all_points.append((name, args, val_str))
        print(f"Parsed: {name}({args}) = {val_str!r}")
    else:
        print(f"No match: {p!r}")

print("\nAll points:")
for pt in all_points:
    print(f"  {pt}")
