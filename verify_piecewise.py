
def check_piecewise():
    print("Verifying AI solution for x=[1, 2, 3, -4], y=[2, 4, 6, -12]")
    
    # Candidate: x + min(x, 2*x)
    # Equivalent to:
    # if x >= 0: min(x, 2x) is x. -> x + x = 2x
    # if x < 0: min(x, 2x) is 2x. -> x + 2x = 3x
    
    data = [(1, 2), (2, 4), (3, 6), (-4, -12)]
    
    for x, y_target in data:
        # Evaluate formula
        term1 = x
        term2 = min(x, 2*x)
        y_pred = term1 + term2
        
        print(f"x={x}: Pred={y_pred}, Target={y_target}. Match? {y_pred == y_target}")

if __name__ == "__main__":
    check_piecewise()
