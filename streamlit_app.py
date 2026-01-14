
import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Note: Both genetic_engine and REPL are imported lazily to reduce startup memory
# This allows the app to load on Streamlit Cloud's limited memory (~1GB)

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

st.title("🧠 Kalkulator AI v1.2-DEBUG (ALTV FIXED)")
st.markdown("### Symbolic Regression Engine")

# --- BROADCAST BANNER ---
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import uuid

# CRITICAL FIX: Disable multiprocessing workers to prevent "MemoryError" / crash in Docker
os.environ["KALKULATOR_ENABLE_PERSISTENT_WORKER"] = "false"
# Clear cache on startup to prevent corruption issues
try:
    cache_dir = Path.home() / ".kalkulator_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
except Exception:
    pass


broadcast_file = os.path.join(os.path.dirname(__file__), "broadcast.txt")
if os.path.exists(broadcast_file):
    try:
        # Use utf-8-sig to handle BOM from Windows PowerShell
        with open(broadcast_file, "r", encoding="utf-8-sig") as f:
            broadcast_msg = f.read().strip()
        if broadcast_msg and not broadcast_msg.startswith("#"):  # Ignore comments
            st.warning(f"📢 **Admin Notice:** {broadcast_msg}")
    except Exception as e:
        st.error(f"Broadcast error: {e}")

# --- SESSION ID FOR PRESENCE ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    
    # Genetic Config (reduced defaults for Streamlit Cloud memory limits)
    pop_size = st.slider("Population Size", 50, 500, 150, step=50)
    generations = st.slider("Generations", 10, 100, 30, step=10)
    patience = st.slider("Patience (Early Stop)", 5, 30, 10)
    
    st.markdown("---")
    
    # LLM Settings
    with st.expander("🤖 AI Tutor Settings"):
        llm_provider = st.selectbox("Provider", ["Google Gemini", "OpenAI (GPT-4)"])
        
        provider_api_key = ""
        selected_model = ""

        # Helper to find best model (defined here to be in scope)
        def get_best_model(api_key):
             if 'detected_auto_model' in st.session_state and st.session_state.detected_auto_model:
                 return st.session_state.detected_auto_model
             
             try:
                 from google import genai
                 client = genai.Client(api_key=api_key)
                 # Get all model names
                 all_models = [m.name.split("/")[-1] for m in client.models.list() if hasattr(m, 'name')]
                 
                 # PRIORITY 0: gemini-1.0-pro (best free tier limits)
                 for m in all_models:
                     if m == "gemini-1.0-pro" or m == "gemini-pro": return m
                 
                 # PRIORITY 1: Versioned Flash (less rate-limited than *-latest)
                 prefers = ["gemini-1.5-flash-002", "gemini-1.5-flash-001", "gemini-1.5-flash"]
                 for p in prefers:
                     if p in all_models: return p
                 
                 # PRIORITY 2: Flash Lite (Newer, usually good but failed recently)
                 for m in all_models:
                     if "flash-lite" in m: return m
                     
                 # PRIORITY 3: Stable Pro 1.5
                 prefers_pro = ["gemini-1.5-pro", "gemini-pro-latest"]
                 for p in prefers_pro:
                     for m in all_models:
                         if p == m: return m

                 # PRIORITY 4: 2.0 / Experimental
                 for m in all_models:
                     if "gemini-2.0-flash" in m: return m
                 # 5. Fallback
                 geminis = [m for m in all_models if "gemini" in m.lower()]
                 if geminis: return geminis[0]
                 return "gemini-1.5-flash"
             except:
                 return "gemini-1.5-flash"
        
        if "Gemini" in llm_provider:
             # Auto-selection feature
             model_options = ["Auto (Best for Key)", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "Custom..."]
             
             model_choice = st.selectbox("Model", model_options)
             
             if model_choice == "Custom...":
                 selected_model = st.text_input("Enter Model Name (e.g. gemini-1.0-pro)", "gemini-1.5-flash")
             elif model_choice == "Auto (Best for Key)":
                 selected_model = "auto"
             else:
                 selected_model = model_choice
             
             # Use secret as default if available
             try:
                 default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
             except (FileNotFoundError, Exception):
                 default_gemini_key = ""

             provider_api_key = st.text_input(
                 "Gemini API Key", 
                 value=default_gemini_key,
                 type="password", 
                 help="Required. Set in Streamlit secrets or enter manually."
             )
             if default_gemini_key:
                 st.caption("✅ Using key from Streamlit secrets")
             else:
                 st.caption("Get a free key at aistudio.google.com")
             
             # Check Available Models button
             if provider_api_key and st.button("🔍 Check Available Models"):
                 try:
                     from google import genai
                     client = genai.Client(api_key=provider_api_key)
                     models = [m.name for m in client.models.list() if hasattr(m, 'name')]
                     gemini_models = [m for m in models if "gemini" in m.lower()]
                     st.success(f"Found {len(gemini_models)} Gemini models:")
                     st.code("\n".join(gemini_models[:15]))  # Show first 15
                 except Exception as e:
                     st.error(f"Error: {e}")
             
        else:
             selected_model = "gpt-4o"
             default_openai_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, 'secrets') else ""
             provider_api_key = st.text_input(
                 "OpenAI API Key", 
                 value=default_openai_key,
                 type="password", 
                 help="Required for OpenAI."
             )
             if default_openai_key:
                 st.caption("✅ Using key from Streamlit secrets")
             else:
                 st.caption("Your key is not stored permanently.")

    st.markdown("---")
    st.markdown("Created by **Syahbana**")
    st.markdown("[https://github.com/sizzlins/kalkulator-ai](https://github.com/sizzlins/kalkulator-ai)")
    
    st.markdown("---")
    
    # --- REPORT ISSUE ---
    # --- REPORT ISSUE ---
    with st.expander("📝 Report Issue / Feedback"):
        st.write("Found a bug or have a feature request? Let us know on GitHub or the Community tab!")
        
        col_gh, col_hf = st.columns(2)
        with col_gh:
            st.link_button("🐛 GitHub Issues", "https://github.com/sizzlins/kalkulator-ai/issues", help="Open a new issue on GitHub")
        with col_hf:
            st.link_button("🤗 HF Community", "https://huggingface.co/spaces/sizzlins/kalkulator-ai/discussions", help="Discuss in the Community tab")
    
    # --- PRESENCE INDICATOR ---
    st.caption(f"Session: `{st.session_state.session_id}`")

# --- MAIN ---

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🖥️ GUI Mode", "⌨️ Terminal Mode", "🤖 AI Tutor"])

# Global Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your math tutor. Run an evolution first, then ask me about the results!"}
    ]

with tab3:
    st.markdown("### 🤖 Math Tutor")
    st.caption("Powered by OpenAI via LangChain (requires API Key)")
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat Input
    if prompt := st.chat_input("Ask about your function..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Check API Key
        if not provider_api_key:
            with st.chat_message("assistant"):
                st.error(f"Please enter your {llm_provider.split()[0]} API Key in the sidebar settings to continue.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Construct Context from Session State (if exists)
                context_str = "No specific result yet."
                if 'last_result_model' in st.session_state:
                     context_str = f"""
                     User has discovered this model: {st.session_state.last_result_model}
                     Error (MSE): {st.session_state.get('last_result_mse', 'N/A')}
                     Complexity: {st.session_state.get('last_result_complexity', 'N/A')}
                     Original Data: {st.session_state.get('last_input_data', 'N/A')}
                     """
                
                system_prompt = f"""You are the AI Tutor for the **Kalkulator AI** web app, a Symbolic Regression tool.

**Your App's UI (Tabs):**
1.  **GUI Mode Tab**: User enters data like `f(1)=3, f(2)=6` in the text box, then clicks "🧬 Evolve Function" to discover a formula.
2.  **Terminal Mode Tab**: A command-line interface for power users (e.g., `plot sin(x)`).
3.  **AI Tutor Tab** (You are here): Answer questions about discovered formulas or how to use the app.

**User's Latest Result (if available):**
{context_str}

**Your Job:**
- If the user asks "how do I use this?", give SPECIFIC steps: "Go to 'GUI Mode', paste the example data, click 'Evolve Function'."
- If they ask about a formula, explain its math simply.
- If they seem confused or ask for examples, OUTPUT DATA USING THIS SPECIAL FORMAT:
  [PREFILL]f(0)=0, f(1)=1, f(2)=8, f(3)=27[/PREFILL]
  This will create a button for the user to auto-fill that data into GUI Mode!
- Be CONCISE. No long textbook explanations.
- Example functions you can suggest:
  - x^2: [PREFILL]f(0)=0, f(1)=1, f(2)=4, f(3)=9, f(4)=16[/PREFILL]
  - sin(x): [PREFILL]f(0)=0, f(1.57)=1, f(3.14)=0, f(4.71)=-1[/PREFILL]
  - Triangle: [PREFILL]f(0)=0, f(0.5)=0.5, f(1)=0, f(1.5)=0.5, f(2)=0[/PREFILL]
  - Chirp: [PREFILL]f(0)=0, f(1.77)=0, f(2.5)=0, f(3.07)=0[/PREFILL]
"""

                try:
                    if "Gemini" in llm_provider:
                        # --- GEMINI LOGIC (New SDK: google-genai) ---
                        from google import genai
                        
                        # Resolve Auto Model
                        final_model_name = selected_model
                        if selected_model == "auto":
                             # We assume get_best_model is defined from sidebar scope
                             # Use status spinner for feedback
                             with st.status("🔍 Auto-detecting best model...", expanded=False) as status:
                                 try:
                                     final_model_name = get_best_model(provider_api_key)
                                     st.session_state.detected_auto_model = final_model_name # Cache
                                     status.update(label=f"Selected: {final_model_name}", state="complete")
                                 except Exception as e:
                                     st.error(f"Auto-detect failed: {e}")
                                     final_model_name = "gemini-1.5-flash"

                        client = genai.Client(api_key=provider_api_key)
                        
                        combined_prompt = f"{system_prompt}\n\nUser Question: {prompt}"
                        
                        try:
                             # Use the resolved model
                            chat = client.chats.create(model=final_model_name)
                            response_stream = chat.send_message_stream(combined_prompt)
                            
                            for chunk in response_stream:
                                if chunk.text:
                                    full_response += chunk.text
                                    message_placeholder.markdown(full_response + "▌")
                                    
                        except Exception as inner_e:
                             err_str = str(inner_e)
                             if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                                 st.error(f"⚠️ Rate Limit Hit for {final_model_name}.")
                                 st.info("Tip: 'gemini-1.5-flash' usually has higher rate limits than Pro or 2.0-Flash.")
                                 st.caption(f"Details: {err_str}")
                             elif "404" in err_str or "NOT_FOUND" in err_str:
                                 st.error(f"⚠️ Model '{final_model_name}' not found for your API Key.")
                                 st.caption("Auto-detect might have picked a region-locked model. Try custom input.")
                             else:
                                 raise inner_e
                                
                    else:
                        # --- OPENAI LOGIC ---
                        import openai
                        client = openai.OpenAI(api_key=provider_api_key)
                        
                        stream = client.chat.completions.create(
                            model="gpt-4o", # Or gpt-3.5-turbo
                            messages=[
                                {"role": "system", "content": system_prompt},
                                *st.session_state.messages
                            ],
                            stream=True,
                        )
                        
                        for chunk in stream:
                             if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                    
                    # Finalize
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # --- PREFILL DETECTION ---
                    import re
                    prefill_match = re.search(r'\[PREFILL\](.*?)\[/PREFILL\]', full_response, re.DOTALL)
                    if prefill_match:
                        prefill_data = prefill_match.group(1).strip()
                        
                        # Strip markers from displayed text, replace with code block
                        clean_response = re.sub(
                            r'\[PREFILL\](.*?)\[/PREFILL\]', 
                            r'```\n\1\n```', 
                            full_response, 
                            flags=re.DOTALL
                        )
                        message_placeholder.markdown(clean_response)
                        # Update stored message too
                        st.session_state.messages[-1]["content"] = clean_response
                        
                        st.info(f"💡 **Suggested data:** `{prefill_data[:50]}...`")
                        if st.button("📋 Use this data in GUI Mode", key=f"prefill_{len(st.session_state.messages)}"):
                            st.session_state.prefill_for_gui = prefill_data
                            st.session_state.gui_input_data = prefill_data
                            st.session_state["gui_textarea_widget"] = prefill_data # Update actual widget key
                            st.toast("✅ Data loaded! Switch to 'GUI Mode' tab now.", icon="📋")
                            st.rerun() # Force page refresh to apply changes
                    
                except Exception as e:
                    st.error(f"AI Provider Error: {e}")

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Data")
        
        input_method = st.radio("Input Method", ["Text Input", "CSV Upload"], horizontal=True)
        
        X_data = None
        y_data = None
        parsed_sucess = False
        
        if input_method == "Text Input":
            # Simple value-based approach: always read from and write to session state
            default_data = "f(1) = 0.841470984807897, f(2) = 0.909297426825682, f(3) = 0.141120008059867, f(4) = -0.756802495307928, f(5) = -0.958924274663138, f(6) = -0.279415498198926"
            
            # Initialize session state if needed
            if 'gui_input_data' not in st.session_state:
                st.session_state.gui_input_data = default_data
            
            # Check if prefill was requested (set by AI Tutor button)
            if 'prefill_for_gui' in st.session_state:
                st.session_state.gui_input_data = st.session_state.prefill_for_gui
                del st.session_state.prefill_for_gui
                st.success("✨ AI-suggested data loaded! Click 'Evolve Function' to discover the formula.")
            
            # Text area with VALUE parameter (not key) - always shows current session state
            user_input = st.text_area(
                "Enter points (e.g., f(0)=1, f(1)=2)", 
                value=st.session_state.gui_input_data, 
                height=150,
                key="gui_textarea_widget"  # Fixed key just for widget identity
            )
            
            # Sync user edits back to session state
            if user_input != st.session_state.gui_input_data:
                st.session_state.gui_input_data = user_input
            
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
                            
                            # Check for function definition: f(x) = expression (where 'x' is in args)
                            # Simple heuristic: if any arg is 'x', 'y' (and not a value), treat as definition
                            is_def = False
                            try:
                                args = [eval_to_float(a.strip()) for a in args_str.split(",")]
                            except:
                                is_def = True
                                
                            if is_def:
                                # Function Definition Mode
                                # f(x) = ...
                                import sympy as sp
                                from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
                                
                                # Parse the expression
                                rhs_str = val_str.strip()
                                input_vars = [a.strip() for a in args_str.split(",")]
                                
                                # Generate synthetic data
                                if len(input_vars) == 1:
                                    # 1D: Generate range
                                    X_synth = np.linspace(-5, 5, 20).reshape(-1, 1)
                                    
                                    # Evaluate expression
                                    local_dict = {v: sp.Symbol(v) for v in input_vars}
                                    # Handle bitwise/custom operators in sympify if needed, or rely on simple cases
                                    # The user input had 'bitwise_xor' which sympy might not handle by default
                                    # We can assume the engine's tools are available or use robust parsing
                                    
                                    # Hack: for complex user expressions like 'bitwise_xor', we might need the engine's eval
                                    # Let's try to pass it to the engine's parser if sympy fails
                                    try:
                                        expr = sp.sympify(rhs_str, locals=local_dict)
                                        tree = ExpressionTree.from_sympy(expr, input_vars)
                                        y_synth = tree.evaluate(X_synth)
                                    except Exception as ex_sympy:
                                        # Fallback to python eval (dangerous but ok for local app) with numpy context
                                        # Need to map custom functions
                                        # For now, just raise or warn
                                        st.warning(f"Could not parse definition symbolically: {ex_sympy}. Trying Python eval...")
                                        try:
                                            # Create safe context with numpy and custom funcs
                                            ctx = {k: getattr(np, k) for k in dir(np)}
                                            ctx.update({"bitwise_xor": np.bitwise_xor, "rshift": np.right_shift}) # Add common bitwise
                                            # Map 'x' to X_synth column
                                            ctx[input_vars[0]] = X_synth[:, 0]
                                            y_synth = eval(rhs_str, {"__builtins__": None}, ctx)
                                        except Exception as ex_eval:
                                            raise ValueError(f"Could not evaluate function: {ex_eval}")
                                            
                                    # Append synthetic data
                                    if x_list == []: # Only if empty so far
                                        X_data = X_synth
                                        y_data = y_synth
                                        x_list.append(True) # Dummy to signal success
                                        st.info("✨ Detected function definition. Generated synthetic training data (20 points).")
                                        break # Stop parsing other lines if definition found
                                else:
                                    st.warning(f"Only 1D function definitions supported for auto-generation for now.")
                                    
                            else:
                                # Normal Data Point
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
                    self.elem.code(full_text[-4000:], language="text")
                    
                    # Also write to original stdout
                    import sys
                    sys.__stdout__.write(message)
                    
                def flush(self):
                    import sys
                    sys.__stdout__.flush()

            with st.spinner("Evolving... (See logs below)"):
                try:
                    # Lazy import to reduce startup memory
                    # Lazy import to reduce startup memory
                    from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
                    from kalkulator_pkg.symbolic_regression.pareto_front import ParetoFront, ParetoSolution 
                    from kalkulator_pkg.function_manager import find_function_from_data
                    import sympy as sp
                    # Import our new Forensic Analysis module
                    from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds
                    
                    # Force reload modules to ensure updates are picked up in long-running Streamlit process
                    import importlib
                    import kalkulator_pkg.function_manager
                    import kalkulator_pkg.function_finder_advanced
                    importlib.reload(kalkulator_pkg.function_finder_advanced)
                    importlib.reload(kalkulator_pkg.function_manager)
                    from kalkulator_pkg.function_manager import find_function_from_data
                    
                    # --- HYBRID MODE: SEEDING ---
                    # Run "Rational Analysis" (find) to get high-quality seeds for rational functions
                    seeds = []
                    try:
                        # Build data list for find(): [(x1,y1), (x2,y2)...]
                        # CRITICAL: Filter out non-finite values (Inf/NaN) for find(), otherwise rational fit fails!
                        find_data = []
                        if X_data is not None and y_data is not None:
                            for i in range(len(y_data)):
                                # Skip Inf/NaN/Complex for find() (Rational Analysis)
                                if not np.isfinite(y_data[i]) or np.iscomplex(y_data[i]):
                                    continue
                                    
                                x_row = tuple(X_data[i]) if X_data.ndim > 1 else (X_data[i],)
                                # Also check inputs
                                if not all(np.isfinite(x) for x in x_row):
                                    continue
                                    
                                find_data.append((x_row, y_data[i]))
                                
                        # Use generic variable names for finding
                        param_chars = "xyzuvwrst"
                        input_vars = [param_chars[i] if i < len(param_chars) else f"x{i+1}" for i in range(X_data.shape[1])]
                        
                        st.info("🧠 Hybrid Mode: Running rational analysis optimization...")
                        # Pass verbose=True to see detection logs in execution log
                        success, func_str, _, note = find_function_from_data(find_data, input_vars, verbose=True)
                        
                        early_exit = False
                        best_sol = None
                        
                        if success and func_str:
                            seeds.append(func_str)
                            if note and "RationalSVD" in str(note):
                                st.success(f"⚡ Rational SVD Discovery: {func_str} ({note})")
                            else:
                                st.info(f"🌱 Rational Seed: {func_str}")
                                
                            # === QUALITY CHECK & EARLY EXIT ===
                            try:
                                from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
                                local_dict = {v: sp.Symbol(v) for v in input_vars}
                                discovered_expr = sp.sympify(func_str, locals=local_dict)
                                tree = ExpressionTree.from_sympy(discovered_expr, input_vars)
                                
                                # Use training data for validation (filtering NaNs/Infs done above)
                                # Re-filter find_data to ensure valid float values
                                y_true = []
                                y_pred = []
                                for (val_x, val_y) in find_data:
                                    try:
                                        pred = float(tree.evaluate(np.array([val_x])))
                                        y_pred.append(pred)
                                        y_true.append(float(val_y))
                                    except:
                                        pass
                                        
                                if len(y_true) > 0:
                                    y_true_arr = np.array(y_true)
                                    y_pred_arr = np.array(y_pred)
                                    mse_val = np.mean((y_true_arr - y_pred_arr)**2)
                                    
                                    st.info(f"🔍 Seed MSE: {mse_val:.2e}")
                                    
                                    # EARLY EXIT THRESHOLD
                                    if mse_val < 0.01:
                                        st.success(f"🎯 Perfect Match Found! Skipping evolution.")
                                        st.balloons()
                                        
                                        # Construct explicit solution to bypass evolution
                                        early_exit = True
                                        best_sol = ParetoSolution(
                                            expression=func_str,
                                            mse=mse_val,
                                            complexity=tree.complexity(),
                                            sympy_expr=discovered_expr,
                                            tree=tree
                                        )
                            except Exception as e:
                                print(f"Seed validation failed: {e}")

                        if not early_exit:
                            # --- PATTERN ANALYSIS (FORENSIC) ---
                            # This enables "Sherlock Mode" for integer patterns like (x-1)/(x+1)
                            st.info("🔍 Forensic Mode: Analyzing patterns (Singularities, Integers)...")
                            pattern_seeds = generate_pattern_seeds(X_data, y_data, variable_names=input_vars, verbose=True)
                            if pattern_seeds:
                                seeds.extend(pattern_seeds)
                                # Show first few seeds in UI
                                display_seeds = pattern_seeds[:3]
                                suffix = f" + {len(pattern_seeds)-3} more" if len(pattern_seeds) > 3 else ""
                                st.info(f"🧬 Forensic Seeds detected: {', '.join(display_seeds)}{suffix}")
                                
                    except Exception as e:
                        print(f"Hybrid seeding failed: {e}")

                    if early_exit and best_sol:
                         # Skip to results
                         pass
                    else:
                        # Configure engine with Hybrid Power
                        config = GeneticConfig(
                            population_size=pop_size * 3, # Boost population for hard problems
                            generations=generations * 3,  # Boost generations
                            patience=patience,
                            verbose=True,
                            seeds=seeds,
                            boosting_rounds=3 # Enable Symbolic Gradient Boosting (matches 'alt' command)
                        )
                        
                        regressor = GeneticSymbolicRegressor(config)

                        regressor = GeneticSymbolicRegressor(config)
                        
                        # DEBUG: Pre-evaluate the seed to verify it works on filtered data
                        if seeds and len(seeds) > 0 and 'X_train' in locals():
                             try:
                                 from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
                                 import sympy as sp
                                 test_seed = seeds[0] # The rational seed is usually first
                                 local_dict = {v: sp.Symbol(v) for v in input_vars}
                                 test_expr = sp.sympify(test_seed, locals=local_dict)
                                 test_tree = ExpressionTree.from_sympy(test_expr, input_vars)
                                 
                                 # Evaluate on filtered data
                                 test_preds = test_tree.evaluate(X_train)
                                 test_diff = test_preds - y_train
                                 test_mse = np.mean(test_diff**2)
                                 
                                 st.info(f"🧪 Seed Validation: '{test_seed}' has MSE={test_mse:.2e} on training data.")
                             except Exception as e:
                                 st.warning(f"⚠️ Seed Validation Warning: Could not evaluate seed '{seeds[0]}': {e}")
                        
                        # Redirect stdout
                        import sys
                        original_stdout = sys.stdout
                        sys.stdout = StreamlitLogger(log_container)
                        
                        try:
                            # Run fit
                            # CRITICAL: Filter out non-finite values (Inf/NaN) from training data
                            # The genetic engine cannot calculate MSE on Infinity.
                            # We used the Infs for Forensic/Rational Analysis (Seeding), but we must hide them for Evolution.
                            filter_mask = np.isfinite(y_data)
                            if not np.all(filter_mask):
                                dropped_count = len(y_data) - np.sum(filter_mask)
                                st.warning(f"⚠️ Filtered {dropped_count} non-finite data points (Infinity/NaN) to allow evolution.")
                                X_train = X_data[filter_mask]
                                y_train = y_data[filter_mask]
                            else:
                                X_train = X_data
                                y_train = y_data
                                
                                X_train = X_data
                                y_train = y_data
                                
                            # ENABLE MULTI-SPACE EVOLUTION (Matches 'alt' command)
                            # This tries evolving in Direct, Log, and Inverse spaces simultaneously
                            st.info("🌌 Multi-Space Mode: Evolving in Direct, Log, and Inverse spaces...")
                            best_expr, best_mse, best_space = regressor.fit_with_transformations(X_train, y_train, input_vars)
                            
                            # Manually construct ParetoFront from best result (fit_with_transformations returns tuple)
                            pareto = ParetoFront()
                            if best_expr:
                                # Calculate complexity
                                try:
                                    from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
                                    symbols = {v: sp.Symbol(v) for v in input_vars}
                                    sympy_expr = sp.sympify(best_expr, locals=symbols)
                                    tree = ExpressionTree.from_sympy(sympy_expr, input_vars)
                                    complexity = tree.complexity()
                                except:
                                    complexity = 10.0 # Fallback
                                    
                                solution = ParetoSolution(
                                    expression=best_expr,
                                    mse=best_mse,
                                    complexity=complexity,
                                    # r2=0.0  <- REMOVED: ParetoSolution does not take this arg
                                    sympy_expr=sympy_expr,  # Added missing required arg
                                    tree=tree               # Added missing required arg
                                )
                                pareto.add(solution)
                                
                                # Show which space won
                                if best_space != "direct":
                                    st.success(f"🚀 Solution found in transformed space: {best_space.upper()}")
                        finally:
                            # Restore stdout
                            sys.stdout = original_stdout
                        
                        st.success("Evolution complete!")
                        
                        # Get best
                        best_sol = pareto.get_best()

                    
                    if best_sol:
                        st.balloons()
                        
                        # Save Context for AI Tutor
                        st.session_state.last_result_model = best_sol.expression
                        st.session_state.last_result_mse = f"{best_sol.mse:.2e}"
                        st.session_state.last_result_complexity = best_sol.complexity
                        st.session_state.last_input_data = user_input if input_method == "Text Input" else "Uploaded CSV data"
                        
                        st.toast("Result found! Go to the 'AI Tutor' tab to ask questions about it ->", icon="🤖")
                        
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
                                    # Evaluate directly using the best solution tree
                                    y_pred_plot = best_sol.tree.evaluate(x_plot)
                                    
                                    # Create dataframe for Altair/Streamlit
                                    # It's easier to use matplotlib for explicit control
                                    import plotly.graph_objects as go
                                    
                                    # Create interactive plot
                                    fig = go.Figure()
                                    
                                    # 1. Scatter Plot for Data (Red Balls)
                                    # Handle complex data for plotting (Project to Real plane)
                                    x_scatter = np.real(X_data).flatten()
                                    y_scatter = np.real(y_data).flatten()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_scatter, 
                                        y=y_scatter,
                                        mode='markers',
                                        name='Data Points (Real Part)',
                                        marker=dict(size=12, color='#ff2b2b', line=dict(width=2, color='white')),
                                        hovertemplate="<b>Input (Re):</b> %{x}<br><b>Target (Re):</b> %{y}<extra></extra>"
                                    ))
                                    
                                    # 2. Line Plot for Discovered Function (Blue Line)
                                    # Ensure we plot real parts only
                                    x_line = np.real(x_plot).flatten()
                                    y_line = np.real(y_pred_plot).flatten()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_line,
                                        y=y_line,
                                        mode='lines',
                                        name=f"Function: {display_expr[:30]}..." if len(display_expr) > 30 else f"f(x) = {display_expr}",
                                        line=dict(color='#0068c9', width=4),
                                        hovertemplate="<b>Prediction (Re):</b> %{y:.4f}<extra></extra>"
                                    ))

                                    # Layout styling for Dark Mode & "Premium Fee"
                                    fig.update_layout(
                                        title=dict(text="Data vs Discovered Model", font=dict(size=20, color='white')),
                                        xaxis=dict(title="Input (x)", showgrid=True, gridcolor='#333', zerolinecolor='#666', fixedrange=True),
                                        yaxis=dict(title="Output (y)", showgrid=True, gridcolor='#333', zerolinecolor='#666', fixedrange=True),
                                        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font=dict(color='white'),
                                        hovermode="closest",  # Focus on the specific point hovered
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom", y=1.02,
                                            xanchor="right", x=1
                                        ),
                                        margin=dict(l=40, r=40, t=40, b=40),
                                        dragmode=False # Disable drag interactions entirely
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
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
    if 'terminal_mode' not in st.session_state:
        st.session_state.terminal_mode = "lite"  # Default to lite mode
    
    # Mode toggle
    st.caption("Enter commands below. Supports math, function finding, and plotting.")
    
    # Init Full Mode defaults
    if 'terminal_mode' in st.session_state:
        del st.session_state.terminal_mode
        
    # Input Form
    with st.form("terminal_form", clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            cli_input = st.text_input("Command >", placeholder="Type help, 1+1, or f(x)=...")
        with col_btn:
            submitted = st.form_submit_button("Run")
    
    if submitted and cli_input:
        import io
        import contextlib
        import sympy as sp
        
        output = ""
        captured_fig = None
        
        # FULL MODE (Standard)
        try:
            # Optimized: Reload modules to support hot-fixes (like 'altv')
            import importlib
            import kalkulator_pkg.cli.repl_core
            import kalkulator_pkg.cli.repl_commands
            # Reload explicit dependencies to ensure new command routing (like 'altv') is active
            importlib.reload(kalkulator_pkg.cli.repl_commands)
            importlib.reload(kalkulator_pkg.cli.repl_core)
            
            # Always re-instantiate REPL to use fresh code/logic
            from kalkulator_pkg.cli.repl_core import REPL
            st.session_state.repl_instance = REPL()
            
            # Restore variables from session state
            if 'cli_vars' in st.session_state and st.session_state.cli_vars:
                 st.session_state.repl_instance.variables = st.session_state.cli_vars.copy()

            repl_instance = st.session_state.repl_instance
            
            # Output buffer
            output_buffer = []
            def capture_output(text):
                output_buffer.append(text)

            # Initialize REPL with callback if not already set or correct it
            # Ensure callback is current (in case of page reload/re-instantiation issues)
            repl_instance.output_callback = capture_output
            
            # Monkey-patch plt.show just to block popups (allow figure capture via gcf)
            original_show = plt.show
            plt.show = lambda: None
            
            is_plot = cli_input.strip().lower().startswith("plot")
            
            # Clean up previous plots
            plt.close('all')
            
            # Capture stdout (for commands that use print() like _handle_evolve)
            with contextlib.redirect_stdout(io.StringIO()) as f:
                # Manual shim for 'altv' if REPL routing is stale
                # 1. Clean input like REPL does (remove > or other prefixes)
                clean_input = cli_input.strip()
                # Remove markdown backticks
                clean_input = clean_input.strip('`')
                # Strip non-alphanumeric prefix chars
                while clean_input and not clean_input[0].isalnum() and clean_input[0] not in '(-+.':
                    clean_input = clean_input[1:]
                
                # DEBUG: Show what the shim sees
                print(f"DEBUG: Input='{cli_input}' Clean='{clean_input}'")
                
                if clean_input.lower().startswith("altv ") or clean_input.lower() == "altv":
                     print("DEBUG: Routing to _handle_evolve via Shim")
                     from kalkulator_pkg.cli.repl_commands import _handle_evolve
                     _handle_evolve(clean_input, repl_instance.variables)
                else:
                    # Run command - output goes to output_buffer via callback (if self.print is used)
                    repl_instance.process_input(cli_input)
            
            std_out = f.getvalue()
            
            if is_plot:
                fig = plt.gcf()
                if fig.get_axes():
                    captured_fig = fig
                    fig.set_size_inches(8, 4)
            
            plt.show = original_show
            
            # Combine callback output and stdout
            output = "".join(output_buffer) + std_out
            # Sync back vars
            st.session_state.cli_vars = repl_instance.variables.copy()

        except MemoryError:
            import psutil
            mem = psutil.virtual_memory()
            output = f"❌ MemoryError: Available: {mem.available/1024/1024:.0f}MB / Total: {mem.total/1024/1024:.0f}MB."
        except Exception as e:
            import traceback
            output = f"Error: {e}\n{traceback.format_exc()}"
        
        # Store in history: 3-tuple (cmd, out, fig)
        st.session_state.cli_history.append((cli_input, output, captured_fig))
        
    # Display History
    st.markdown("---")
    
    # Feature: Send Output to GUI
    # If the latest output contains "f(...) = ...", offer to send it to GUI input
    if st.session_state.cli_history:
        # Handle unpacking safely (old history might be 2-tuple)
        last_item = st.session_state.cli_history[-1]
        if len(last_item) == 3:
            _, last_out, _ = last_item
            import re
            # Check for data pattern: f(number) = number
            if re.search(r"f\([^)]+\)\s*=\s*", str(last_out)):
                col_hist_info, col_send_btn = st.columns([3, 1])
                with col_send_btn:
                    # Callback to safely update session state before rerun
                    def send_to_gui_callback(output_text):
                        clean_lines = []
                        for line in str(output_text).split('\n'):
                            if "=" in line and "(" in line:
                                clean_lines.append(line.strip())
                        
                        if clean_lines:
                            data = ", ".join(clean_lines)
                        else:
                            data = str(output_text).strip()
                        
                        st.session_state.gui_input_data = data
                        st.session_state["gui_textarea_widget"] = data
                        st.toast("✅ Data sent to GUI Mode! Switch tabs to evolve.", icon="🚀")

                    st.button(
                        "📋 Send Last Output to GUI", 
                        key="send_last_to_gui", 
                        help="Copy this data to the GUI Mode input box",
                        on_click=send_to_gui_callback,
                        args=(last_out,)
                    )
    # Loop history
    for item in reversed(st.session_state.cli_history):
        # Handle backward compatibility if tuple length changed (old history)
        if len(item) == 2:
            cmd, out = item
            fig = None
        else:
            cmd, out, fig = item
            
        st.markdown(f"**> {cmd}**")
        if fig:
            st.pyplot(fig, use_container_width=False)
        st.code(out)

