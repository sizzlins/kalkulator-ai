# Engineering Standards (The Kalkulator Constitution)

**Rules for safety-critical systems.**

1.  **Simple Control Flow**: Avoid complex "magic" (metaclasses, dynamic attribute injection, `exec`). Keep logic linear.
2.  **Bounded Loops**: All loops (especially in genetic programming and numerical solvers) must have a fixed `max_iterations` failsafe.
3.  **Object Stability**: Avoid runtime structure modification ("monkey patching"). Treat initialized objects as immutable where possible.
4.  **Small Units**: Functions should fit on a single screen (~60 lines). If larger, refactor.
5.  **Defensive Design**: Minimum 2 assertions per function. Validate assumptions (e.g., `assert x > 0`) to catch impossible states early.
6.  **Encapsulation**: Minimize global state. Keep variables scoped as locally as possible.
7.  **Explicit Error Handling**: Never swallow exceptions (`try: ... except: pass`). Handle failures explicitly or let them crash visibly.
8.  **No "Magic" Preprocessing**: Limit complex decorators that obscure function signatures. Code should be transparent.
9.  **Shallow Nesting**: Avoid deep nesting of `if/else/for`. Flatten logic to reduce cyclomatic complexity.
10. **Zero Tolerance for Warnings**: Treat strict `mypy` errors and `ruff` warnings as blocking bugs.
11. **Deep Assessment (The "Pondering" Rule)**: Before writing a single line of code, we MUST:
    - 100% understand the Root Cause (not just the symptom).
    - Plan the fix in detail.
    - "Double Think": Critically evaluate if the fix is actually a good idea or just a band-aid.
    - Only then, execute.
12. **Future-Proofing (The "Anti-Dev-Hell" Rule)**: Always design for the future. Ask: "Will this change make life miserable for the next developer?" If yes, don't do it. Avoid shortcuts that lead to technical debt or "development hell." Decouple concerns (e.g., calculation vs. presentation) to ensure long-term maintainability.
13. **The Final REPL Check**: After all edits are complete, you MUST manually test the REPL using `kalkulator.py` (not just unit tests). This is the user's entry point. Run varied inputs to verify you haven't broken the interactive experience. 'Does it actually work for the user?' is the final exam.
14. **Rule Recitation Protocol**: Before beginning any significant task, you MUST recite all these rules if you do not remember them. This ensures they remain top-of-mind and active during development.
