
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
                    # Lazy import to reduce startup memory
                    from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
                    
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
    if 'terminal_mode' not in st.session_state:
        st.session_state.terminal_mode = "lite"  # Default to lite mode
    
    # Mode toggle
    col_mode, col_info = st.columns([1, 3])
    with col_mode:
        terminal_mode = st.radio(
            "Mode",
            ["lite", "full"],
            index=0 if st.session_state.terminal_mode == "lite" else 1,
            horizontal=True,
            help="Lite = fast, Full = heavy (may crash on cloud)"
        )
        st.session_state.terminal_mode = terminal_mode
    with col_info:
        if terminal_mode == "lite":
            st.caption("🚀 **Lite Mode**: Fast sympy-based evaluator")
        else:
            st.caption("⚠️ **Full Mode**: Uses heavy REPL (may cause MemoryError on cloud)")
        
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
        
        # Check if FULL mode is enabled - use the heavy REPL
        if st.session_state.terminal_mode == "full":
            try:
                from kalkulator_pkg.cli.repl_core import REPL
                
                repl_instance = REPL()
                repl_instance.variables = st.session_state.cli_vars
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    # Monkey-patch plt.show
                    original_show = plt.show
                    plt.show = lambda: None
                    
                    is_plot = cli_input.strip().lower().startswith("plot")
                    repl_instance.process_input(cli_input)
                    
                    if is_plot:
                        fig = plt.gcf()
                        if fig.get_axes():
                            captured_fig = fig
                            fig.set_size_inches(8, 4)
                    
                    plt.show = original_show
                
                output = f.getvalue()
                st.session_state.cli_vars = repl_instance.variables
            except MemoryError:
                output = "❌ MemoryError: Switch to Lite Mode or reduce settings."
            except Exception as e:
                output = f"Error: {e}"
        else:
            # LITE MODE - lightweight sympy evaluator
            try:
                # Check for special commands first
                cmd_lower = cli_input.strip().lower()
            
                if cmd_lower == "help":
                    output = """Kalkulator AI v1.5.0 (Terminal Mode)

BASIC MATH
  1+1, 2*3, sin(pi/2)     Evaluate expressions
  x = 5                   Define variable
  
FUNCTION DEFINITION
  f(x) = x^2              Define function
  f(1), f(2), f(3)        Call function

EQUATION SOLVING
  x^2+x-6=0               Solve for x → x = 2, -3

FUNCTION DISCOVERY
  find f(1)=1, f(2)=4, f(3)=9    Discover f(x) from data

PLOTTING
  plot sin(x)             Plot a function
"""
                # Handle FIND command - function discovery
                elif cmd_lower.startswith("find "):
                    data_str = cli_input[5:].strip()
                    # Parse f(x)=y pairs
                    import re
                    pairs = re.findall(r'f\(([^)]+)\)\s*=\s*([^\s,]+)', data_str)
                    if pairs:
                        # Lazy import to reduce startup memory
                        from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
                        
                        X_data = np.array([[float(x)] for x, y in pairs])
                        y_data = np.array([float(y) for x, y in pairs])
                        
                        config = GeneticConfig(
                            population_size=pop_size,
                            generations=generations,
                            patience=patience
                        )
                        regressor = GeneticSymbolicRegressor(config)
                        
                        # Capture output
                        import io, contextlib
                        f = io.StringIO()
                        with contextlib.redirect_stdout(f):
                            front = regressor.fit(X_data, y_data, variables=["x"])
                        
                        if front and front.solutions:
                            best = front.get_best()
                            output = f"Found: f(x) = {best.expression}\nMSE: {best.mse:.6e}"
                        else:
                            output = "No function found."
                    else:
                        output = "Usage: find f(1)=1, f(2)=4, f(3)=9"
                    
                # Handle PLOT command
                elif cmd_lower.startswith("plot "):
                    expr_str = cli_input[5:].strip()
                    try:
                        expr = sp.sympify(expr_str)
                        x_sym = sp.Symbol('x')
                        f_lambda = sp.lambdify(x_sym, expr, modules=['numpy'])
                        
                        x_vals = np.linspace(-10, 10, 200)
                        y_vals = f_lambda(x_vals)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                        ax.grid(True, alpha=0.3)
                        ax.set_title(f"y = {expr_str}")
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        
                        # Dark theme
                        ax.set_facecolor('#0e1117')
                        fig.patch.set_facecolor('#0e1117')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                        
                        captured_fig = fig
                        output = f"Plotted: y = {expr_str}"
                    except Exception as e:
                        output = f"Plot error: {e}"
                elif "=" in cli_input and not cli_input.strip().startswith("="):
                    # Check if it's an equation to solve (lhs = number or lhs = 0)
                    parts = cli_input.split("=", 1)
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    
                    import re
                    
                    # Check if function definition: f(x) = ...
                    func_match = re.match(r'(\w+)\(([^)]+)\)', lhs)
                    if func_match:
                        name = func_match.group(1)
                        args = func_match.group(2)
                        st.session_state.cli_vars[name] = {"args": args, "expr": rhs}
                        output = f"Function '{name}' defined."
                    # Check if equation solving: expr = 0 or expr = number
                    elif rhs == "0" or re.match(r'^-?\d+\.?\d*$', rhs):
                        # This is an equation to solve
                        try:
                            # Move rhs to lhs: lhs - rhs = 0
                            if rhs != "0":
                                equation = f"({lhs}) - ({rhs})"
                            else:
                                equation = lhs
                            
                            expr = sp.sympify(equation)
                            # Find all free symbols (variables)
                            symbols = list(expr.free_symbols)
                            if symbols:
                                solutions = sp.solve(expr, symbols[0])
                                if solutions:
                                    sol_str = ", ".join([str(s) for s in solutions])
                                    output = f"{symbols[0]} = {sol_str}"
                                else:
                                    output = "No solution found."
                            else:
                                output = "No variable to solve for."
                        except Exception as e:
                            output = f"Solve error: {e}"
                    else:
                        # Variable assignment
                        try:
                            val = sp.sympify(rhs).evalf()
                            st.session_state.cli_vars[lhs] = float(val)
                            output = f"{lhs} = {val}"
                        except:
                            output = f"Error parsing: {rhs}"
                else:
                    # Expression evaluation
                    import re
                    
                    # Helper function to evaluate a single expression
                    def eval_single_expr(expr_str):
                        # Substitute function calls: f(1) -> evaluate the stored function
                        def eval_func_call(match):
                            fname = match.group(1)
                            fargs = match.group(2)
                            if fname in st.session_state.cli_vars:
                                func_def = st.session_state.cli_vars[fname]
                                if isinstance(func_def, dict) and "args" in func_def:
                                    # It's a function - substitute args into expr
                                    param_names = [p.strip() for p in func_def["args"].split(",")]
                                    arg_vals = [a.strip() for a in fargs.split(",")]
                                    func_expr = func_def["expr"]
                                    for pname, aval in zip(param_names, arg_vals):
                                        func_expr = re.sub(rf'\b{pname}\b', f'({aval})', func_expr)
                                    return f"({func_expr})"
                            return match.group(0)  # Return unchanged if not found
                        
                        # Find all function calls like f(1) or g(2, 3)
                        expr_str = re.sub(r'(\w+)\(([^)]+)\)', eval_func_call, expr_str)
                        
                        # Substitute numeric variables
                        for var, val in st.session_state.cli_vars.items():
                            if isinstance(val, (int, float)):
                                expr_str = re.sub(rf'\b{var}\b', str(val), expr_str)
                        
                        return sp.sympify(expr_str).evalf()
                    
                    # Split by commas NOT inside parentheses (top-level commas only)
                    # Simple approach: split by ", " (comma-space pattern for chains)
                    parts = [p.strip() for p in re.split(r',\s*(?![^()]*\))', cli_input)]
                    
                    if len(parts) > 1:
                        # Multiple expressions - evaluate each
                        results = []
                        for part in parts:
                            try:
                                results.append(str(eval_single_expr(part)))
                            except Exception as e:
                                results.append(f"Error: {e}")
                        output = ", ".join(results)
                    else:
                        # Single expression
                        result = eval_single_expr(cli_input)
                        output = str(result)
                    
            except Exception as e:
                output = f"Error: {e}"
        
        # Variables are already in st.session_state.cli_vars - no need to sync
        
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

