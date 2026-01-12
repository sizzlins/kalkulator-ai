
import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

# Import REPL at module level (once at startup, not per-command)
from kalkulator_pkg.cli.repl_core import REPL

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

# --- BROADCAST BANNER ---
import os
import json
from datetime import datetime
import uuid

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
             default_gemini_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, 'secrets') else ""
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
    with st.expander("📝 Report Issue / Feedback"):
        report_text = st.text_area("Describe the issue or feedback:", height=100, key="report_text")
        report_email = st.text_input("Your email (optional):", key="report_email")
        
        if st.button("Submit Report", key="submit_report"):
            if report_text.strip():
                try:
                    # Try to send email if SMTP secrets are configured
                    smtp_configured = False
                    if hasattr(st, 'secrets'):
                        smtp_email = st.secrets.get("SMTP_EMAIL", "")
                        smtp_password = st.secrets.get("SMTP_PASSWORD", "")
                        admin_email = st.secrets.get("ADMIN_EMAIL", smtp_email)
                        smtp_configured = bool(smtp_email and smtp_password)
                    
                    if smtp_configured:
                        import smtplib
                        from email.mime.text import MIMEText
                        
                        subject = f"[Kalkulator AI] Report from {st.session_state.session_id}"
                        body = f"""
New Report from Kalkulator AI
=============================
Session ID: {st.session_state.session_id}
Timestamp: {datetime.now().isoformat()}
User Email: {report_email.strip() if report_email else 'Not provided'}

Message:
{report_text.strip()}
"""
                        msg = MIMEText(body)
                        msg['Subject'] = subject
                        msg['From'] = smtp_email
                        msg['To'] = admin_email
                        
                        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                            server.login(smtp_email, smtp_password)
                            server.sendmail(smtp_email, admin_email, msg.as_string())
                        
                        st.success("✅ Report sent! Thank you for your feedback.")
                    else:
                        # Fallback: show the report for manual copy
                        st.success("✅ Report received! (Email not configured)")
                        st.code(f"Session: {st.session_state.session_id}\nMessage: {report_text.strip()}")
                        
                except Exception as e:
                    st.error(f"Failed to send: {e}")
            else:
                st.warning("Please enter a message.")
    
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
  - x^3: [PREFILL]f(0)=0, f(1)=1, f(2)=8, f(3)=27[/PREFILL]
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
                            st.session_state.gui_input_text = prefill_data # Directly set the key
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
        # REPL is imported at module level to avoid repeated heavy imports
        # Initialize REPL with session variables
        repl_instance = REPL()
        repl_instance.variables = st.session_state.cli_vars
        
        import io
        import contextlib
        import matplotlib.pyplot as plt
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                # Monkey-patch plt.show to avoid popping windows on server
                original_show = plt.show
                plt.show = lambda: None
                
                # Check for plot command
                is_plot = cli_input.strip().lower().startswith("plot")
                
                # process_input handles commands, help, AND math expressions
                repl_instance.process_input(cli_input)
                
                # If command was plot, check for active figure
                captured_fig = None
                if is_plot:
                   fig = plt.gcf()
                   if fig.get_axes(): # Only if axes were drawn
                       captured_fig = fig # Store for history
                       fig.set_size_inches(8, 4) # Resize for web (smaller than default 10x6)
                       # st.pyplot(fig) # Removed to avoid duplicate (handled by history loop)
                       pass 
                
                # Restore original show
                plt.show = original_show
                
            except Exception as e:
                print(f"Error: {e}")
                
        output = f.getvalue()
        
        # Sync variables back
        st.session_state.cli_vars = repl_instance.variables
        
        # Store in history: 3-tuple (cmd, out, fig)
        # Note: fig object is mutated (resized), so history will use new size
        st.session_state.cli_history.append((cli_input, output, captured_fig))
        
    # Display History
    st.markdown("---")
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

