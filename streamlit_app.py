
import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

# Page config
st.set_page_config(
    page_title="Kalkulator AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #fffffe;
    }
    .stButton>button {
        background-color: #7928d2;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #924ce0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Kalkulator AI")
st.markdown("### Symbolic Regression Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    
    # Genetic Config
    pop_size = st.slider("Population Size", 100, 1000, 300, step=50)
    generations = st.slider("Generations", 10, 200, 50, step=10)
    patience = st.slider("Patience (Early Stop)", 5, 50, 10)
    
    st.markdown("---")
    st.markdown("Created by **Syahbana**")
    st.markdown("[https://github.com/sizzlins/kalkulator-ai](https://github.com/sizzlins/kalkulator-ai)")

# --- MAIN ---

# --- TABS ---
tab1, tab2 = st.tabs(["🖥️ GUI Mode", "⌨️ Terminal Mode"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Data")
        
        input_method = st.radio("Input Method", ["Text Input", "CSV Upload"], horizontal=True)
        
        X_data = None
        y_data = None
        parsed_sucess = False
        
        if input_method == "Text Input":
            default_text = "f(0)=0, f(1)=0.8415, f(2)=0.9093, f(3)=0.1411, f(4)=-0.7568, f(5)=-0.9589" # sin(x)
            user_input = st.text_area("Enter points (e.g., f(0)=1, f(1)=2)", default_text, height=150)
            
            if user_input:
                # Parse regex like CLI
                # Matches f(args)=val
                # Handles f(1, 2) = 3
                pts = []
                
                # Normalize
                text = user_input.replace("\n", ",")
                parts = [p.strip() for p in text.split(",") if p.strip()]
                
                x_list = []
                y_list = []
                
                try:
                    # Import robust parser
                    from kalkulator_pkg.utils.parsing import eval_to_float
                    
                    for part in parts:
                        # Regex: match name(args)=val
                        match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*=\s*([^=]+)$", part)
                        if match:
                            args_str = match.group(2)
                            val_str = match.group(3)
                            
                            # Parse args using robust eval
                            args = [eval_to_float(a.strip()) for a in args_str.split(",")]
                            val = eval_to_float(val_str)
                            
                            x_list.append(args)
                            y_list.append(val)
                    
                    if x_list:
                        X_data = np.array(x_list)
                        y_data = np.array(y_list)
                        parsed_sucess = True
                        st.success(f"Parsed {len(y_data)} data points.")
                    else:
                        st.warning("No valid data points found. Format: f(x)=y")
                        
                except Exception as e:
                    st.error(f"Parsing error: {e}")

        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview:", df.head())
                    
                    # Assume last column is target y, others are X
                    X_data = df.iloc[:, :-1].values
                    y_data = df.iloc[:, -1].values
                    parsed_sucess = True
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

    # --- ACTION ---

    with col1: # Put button below input
        if parsed_sucess and st.button("🧬 Evolve Function", use_container_width=True):
            
            # Create a placeholder for logs
            st.markdown("### 📜 Execution Logs")
            log_container = st.empty()
            
            # Custom Logger to redirect stdout to Streamlit
            class StreamlitLogger(object):
                def __init__(self, elem):
                    self.elem = elem
                    self.log_history = []
                    
                def write(self, message):
                    # Filter out purely empty newlines to save space if needed
                    # but keeping format is better.
                    
                    # Optional: Standardize notation in logs too
                    if 'X_data' in locals() or 'X_data' in globals():
                         # We are inside the function where X_data is defined, 
                         # but to be safe we can just blindly replace if we know it's 1D context.
                         # Actually, simpler: create StreamlitLogger with 'is_1d' flag.
                         pass

                    # Just hard replace x0 with x for now in logs if it looks like math?
                    # Or rely on scope.
                    # Since this class is defined inside the block where X_data exists:
                    if X_data is not None and len(X_data.shape) > 1 and X_data.shape[1] == 1:
                        message = message.replace("x0", "x")

                    self.log_history.append(message)
                    # Show last 30 lines to keep UI snappy
                    full_text = "".join(self.log_history)
                    # Use code block for monospaced log look
                    self.elem.code(full_text[-3000:], language="text")
                    
                    # Also write to original stdout
                    import sys
                    sys.__stdout__.write(message)
                    
                def flush(self):
                    import sys
                    sys.__stdout__.flush()

            with st.spinner("Evolving... (See logs below)"):
                try:
                    # Configure engine
                    config = GeneticConfig(
                        population_size=pop_size,
                        generations=generations,
                        patience=patience,
                        verbose=True
                    )
                    
                    regressor = GeneticSymbolicRegressor(config)
                    
                    # Redirect stdout
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = StreamlitLogger(log_container)
                    
                    try:
                        # Run fit
                        pareto = regressor.fit(X_data, y_data)
                    finally:
                        # Restore stdout
                        sys.stdout = original_stdout
                    
                    st.success("Evolution complete!")
                    
                    # Get best
                    best_sol = pareto.get_best()
                    
                    if best_sol:
                        st.balloons()
                        
                        # Show Result
                        with col2:
                            st.subheader("2. Results")
                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.markdown("### 🎯 Best Result")
                                
                                # Sanitize for display: x0 -> x if 1D
                                display_expr = best_sol.expression
                                if X_data.shape[1] == 1:
                                    display_expr = display_expr.replace("x0", "x")
                                
                                st.latex(f"f(x) = {display_expr}".replace("**", "^").replace("*", ""))
                                st.code(display_expr, language="python")
                                
                            with res_col2:
                                st.metric("MSE (Error)", f"{best_sol.mse:.2e}")
                                st.metric("Complexity", f"{best_sol.complexity}")
                            
                            # --- VISUALIZATION ---
                            st.markdown("### 📈 Visualization")
                            
                            # Generate plot data
                            if X_data.shape[1] == 1:
                                x_plot = np.linspace(X_data.min(), X_data.max(), 200).reshape(-1, 1)
                                
                                # Evaluate on dense grid
                                try:
                                    y_pred_plot = regressor.predict(x_plot)
                                    
                                    # Create dataframe for Altair/Streamlit
                                    # It's easier to use matplotlib for explicit control
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    ax.scatter(X_data, y_data, color='red', label='Data Points', zorder=5)
                                    ax.plot(x_plot, y_pred_plot, color='blue', label='Discovered: ' + display_expr[:30] + '...', linewidth=2)
                                    ax.grid(True, alpha=0.3)
                                    ax.legend()
                                    ax.set_title("Data vs Model")
                                    
                                    # Style
                                    ax.set_facecolor('#0e1117')
                                    fig.patch.set_facecolor('#0e1117')
                                    ax.tick_params(colors='white')
                                    ax.xaxis.label.set_color('white')
                                    ax.yaxis.label.set_color('white')
                                    ax.spines['top'].set_color('white')
                                    ax.spines['bottom'].set_color('white')
                                    ax.spines['left'].set_color('white')
                                    ax.spines['right'].set_color('white')
                                    # Legend text
                                    plt.setp(ax.get_legend().get_texts(), color='black') # Matplotlib legend is usually white bg
                                    
                                    st.pyplot(fig)
                                    
                                except Exception as e:
                                    st.error(f"Plotting error: {e}")
                            else:
                                st.info("Top-down heatmap visualization not implemented for >1D data yet.")
                            
                    else:
                        st.error("No solution found.")
                        
                except Exception as e:
                    # Restore stdout in case of error
                    import sys
                    sys.stdout = sys.__stdout__
                    st.error(f"Engine Error: {e}")
                    st.exception(e)

with tab2:
    st.markdown("### ⌨️ Terminal")
    st.markdown("Execute raw CLI commands directly.")
    
    # Initialize session state for CLI
    if 'cli_history' not in st.session_state:
        st.session_state.cli_history = []
    if 'cli_vars' not in st.session_state:
        st.session_state.cli_vars = {}
        
    # Input
    # Input Form
    with st.form("terminal_form", clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            cli_input = st.text_input("Command >", placeholder="Type help, 1+1, or f(x)=...")
        with col_btn:
            submitted = st.form_submit_button("Run")
    
    if submitted and cli_input:
        # Import the full REPL core
        from kalkulator_pkg.cli.repl_core import REPL
        
        # Initialize REPL with session variables
        repl_instance = REPL()
        repl_instance.variables = st.session_state.cli_vars
        
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                # process_input handles commands, help, AND math expressions
                repl_instance.process_input(cli_input)
            except Exception as e:
                print(f"Error: {e}")
                
        output = f.getvalue()
        
        # Sync variables back
        st.session_state.cli_vars = repl_instance.variables
        
        st.session_state.cli_history.append((cli_input, output))
        
    # Display History
    st.markdown("---")
    for cmd, out in reversed(st.session_state.cli_history):
        st.markdown(f"**> {cmd}**")
        st.code(out)

