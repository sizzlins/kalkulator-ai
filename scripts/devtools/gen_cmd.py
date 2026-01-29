
import math

parts = []
# Grid -5 to 5
for x in range(-5, 6):
    for y in range(-5, 6):
        val = x**2 - y**2
        parts.append(f"f({x}, {y}) = {val}")

# Special Points (from user log)
parts.append("f(-20, -19) = 39")
parts.append("f(20, e) = 392.61094390106933")
parts.append("f(pi, i) = -i^2 + pi^2")
parts.append("f(sin(1), sin(pi)) = 0.7080734182735712") 
parts.append("f(4.1, -2.5) = 10.559999999999999")
parts.append("f(cos(0), 1+2i) = 1 - (2i + 1)^2")

print("altvd " + ", ".join(parts))
