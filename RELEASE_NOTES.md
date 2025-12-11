# Release Notes

## v1.2.0 (2025-12-11)

### Features

- **LambertW Discovery**: Engine now identifies `exp(LambertW(log(x)))` (Inverse of `x^x`) and other LambertW patterns.
- **Absolute Mode**: Documentation rewritten for conciseness and clarity.

### Fixes

- **REPL Parse Error**: Resolved syntax error when persisting functions with low confidence scores.
- **Version Display**: CLI banner now shows correct version number.

## v1.1.0 (2025-12-09)

### Features

- **Persistence**: Commands `save`, `loadfunction`, `clearfunction` added.
- **Symbolic Constants**: Detection of `pi`, `e` in outputs.

### Fixes

- **Physics Boosting**: Tuned penalties for relativistic intervals.

## v1.0.0 (2025-12-06)

### Features

- **Symbolic Regression**: Sparse regression, GP, SINDy.
- **Physics Engine**: Detection of Inverse Square, Euclidean, Harmonic patterns.
- **Calculator**: Solver, Calculus, Plotting.
