
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
    st.markdown("Created by **Kalkulator Team**")

# --- MAIN ---

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
                for part in parts:
                    # Regex: match name(args)=val
                    match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*=\s*([^=]+)$", part)
                    if match:
                        args_str = match.group(2)
                        val_str = match.group(3)
                        
                        # Parse args
                        args = [float(a.strip()) for a in args_str.split(",")]
                        val = float(val_str)
                        
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

if parsed_sucess and st.button("🧬 Evolve Function", use_container_width=True):
    with st.spinner("Evolving... (This may take a moment)"):
        try:
            # Configure engine
            config = GeneticConfig(
                population_size=pop_size,
                generations=generations,
                patience=patience,
                verbose=True
            )
            
            regressor = GeneticSymbolicRegressor(config)
            
            # Status placeholder
            status_text = st.empty()
            status_text.text("Initializing population...")
            
            # Capture stdout? Hard in streamlit. Just run it.
            pareto = regressor.fit(X_data, y_data)
            
            status_text.text("Evolution complete!")
            
            # Get best
            best_sol = pareto.get_best()
            
            if best_sol:
                st.balloons()
                
                # Show Result
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown("### 🎯 Best Result")
                    st.latex(f"f(x) = {best_sol.expression}".replace("**", "^").replace("*", ""))
                    st.code(best_sol.expression, language="python")
                    
                with res_col2:
                    st.metric("MSE (Error)", f"{best_sol.mse:.2e}")
                    st.metric("Complexity", f"{best_sol.complexity}")
                
                # --- VISUALIZATION ---
                st.markdown("### 📈 Visualization")
                
                # Generate plot data
                # If 1D, plot line. If 2D, plot ??? (Heatmap maybe? skipping for now, assume 1D mostly)
                if X_data.shape[1] == 1:
                    x_plot = np.linspace(X_data.min(), X_data.max(), 200).reshape(-1, 1)
                    
                    # Evaluate on dense grid
                    try:
                        y_pred_plot = regressor.predict(x_plot)
                        
                        # Create dataframe for Altair/Streamlit
                        # It's easier to use matplotlib for explicit control
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.scatter(X_data, y_data, color='red', label='Data Points', zorder=5)
                        ax.plot(x_plot, y_pred_plot, color='blue', label='Discovered: ' + best_sol.expression[:30] + '...', linewidth=2)
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
            st.error(f"Engine Error: {e}")
            st.exception(e)
