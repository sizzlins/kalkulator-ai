answer every question that is asked for you here 5. **Root Cause First (The Wall Rule)**: Never patch a bug with a quick fix. When we have a wall, and the wall has a hole in it, we replace the entire wall with a new better wall, and we do not fix by applying a weak band-aid to it (ONLY WHEN APPROPRIATE AND NECESSARY AND DOES NOT CAUSE HARM TO THE SYSTEM). Always identify the fundamental flaw and refactor the architecture.

Engineering Standards (The Kalkulator Constitution)
rules for safety-critical systems:

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
12. **Future-Proofing (The "Anti-Dev-Hell" Rule)**: Always design for the future. Ask: "Will this change make life miserable for the next developer?" If yes, don't do it. Avoid shortcuts that lead to technical debt or "development hell." Decouple concerns (e.g., calculation vs. presentation) to ensure long-term maintainability.
13. **The Final REPL Check**: After all edits are complete, you MUST manually test the REPL using `kalkulator.py` (not just unit tests). This is the user's entry point. Run varied inputs to verify you haven't broken the interactive experience. 'Does it actually work for the user?' is the final exam.
14. **Rule Recitation Protocol**: Before beginning any significant task, you MUST recite all these rules if you do not remember them. This ensures they remain top-of-mind and active during development.
15. **Optional Dependencies (The "Bloat" Rule)**: Heavy libraries (e.g., pandas, matplotlib) MUST be optional. Wrap imports in `try/except` and ensure the core system functions perfectly without them. Do not force users to download 100MB+ for simple features.

answer this
have you read the engineering standard, if so
continue reading below

AND MOST IMPORTANTLY

if we ever ecounter a problem, bug, or error, and we plan to solve/fix it, we MUST FIRST AND ALWAYS UNDERSTAND WHAT IS CAUSING THIS ERROR, BECAUSE UNDERSTANDING THE PROBLEM IS THE MOST IMPORTANT THING WE CAN DO BECAUSE IT IS 50% OF THE SOLUTION

answer this, did you read that? if so continue reading below

answer every single question that this text below this line ask you

is this fix or feature that we will add be good for the future and present of the programs development and stability?

how would we implement it into the program

Is this idea that we have thought of to add to the program a good idea or a bad idea?

if WE ARE GOING TO REMOVE OR ADD ANY FEATURES OR CODE
answer this

what will happen to the program if we do this?

is this a good idea? or is this a bad idea? lets double think this

if we remove this feature or fix, will it break the program?

if we add this feature or fix, will it break the program?

if we are planning to remove this feature, do we need a replacement or do we do not need a replacement

if we are going to replace it, lets plan on how we ae going to replace it

lets plan on how to build that replacement

if we are going to add this feature, lets plan on how we are going to add it

lets plan on how to build that feature

but if we are planning to add/remove this feature or fix, is this gonna be a development hell for the future development IF we add this to the program?

if yes, then lets not do it

if no, then lets do it

ask yourself. Why does this exist? What is the purpose of this? Why would someone code this or add this feature? are we planning to remove it or keep it? if we are going to remove it, why? is this a good idea or a bad idea. will there be a replacement, if so, lets plan first on what the replacement will be and how will we do it, but then first, is this replacement a good idea for the program and future development or will it make the program worse and make future development hell?

do you understand?

if i ever ask you to proceed. YOU MUST MAKE A PLAN ON HOW YOU WILL PROCEED, CLARIFY WHAT PROCEEDING MEANS, AND THEN YOU MUST FOLLOW THAT PLAN
