
# Release Notes V1.0.0: Golden Master Stable

## 🚀 Major Milestone
Kalkulator AI has reached Version 1.0.0. This release marks the transition from Beta to a **feature-complete, stable scientific tool**.

## ✨ Key Features
*   **Robust Symbolic Regression:** Discovers mathematical laws from data points (Polynomials, Rationals, Trigonometric, Exponentials).
*   **Physics Law Detection:** Specialized handling for:
    *   Inverse Square Laws (`m/r^2`)
    *   Euclidean Distances (`sqrt(a^2+b^2)`)
    *   Geometric Means (`sqrt(x*y)`)
    *   Harmonic Means / Resistors (`xy/(x+y)`)
    *   Relativistic Factors (`1/sqrt(1-v^2)`)
    *   Difference of Squares (`a^2-b^2`)
*   **Stability Guarantee:**
    *   Safe Division handling for singularities (0/0, division by zero).
    *   Complex root protection (negative domains masked).
*   **Full Calculator Suite:**
    *   Equation Solving (Linear, Quadratic, Systems).
    *   Calculus (Differentiation, Integration).
    *   Plotting (2D graphs via matplotlib).

## 🛠 Fixes & Improvements
*   Fixed "Lazy Linear" shortcuts where the engine would prefer fitting a line to sparse data instead of the true curve.
*   Implemented explicit pre-checks for 2-variable common physics patterns.
*   Added "Golden Master" verification suite to prevent regression.

## 📦 Installation
```bash
pip install kalkulator
```
or from source:
```bash
git clone https://github.com/sizzlins/kalkulator-ai.git
cd kalkulator-ai
pip install .
```
