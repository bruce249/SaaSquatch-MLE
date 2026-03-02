import json
import os
import time

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

from icp_engine import DATA_FILE, EXAMPLE_PROMPT, run_icp_pipeline

# Streamlit App Configuration
st.set_page_config(
    page_title="Hybrid ICP Scoring Engine",
    page_icon="🎯",
    layout="wide",
)

# Helper Functions
@st.cache_data
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        companies = json.load(f)
    return pd.DataFrame(companies)

def run_analysis(user_prompt, hf_token, df):
    hf_client = InferenceClient(token=hf_token)
    
    with st.spinner("Pipeline Running..."):
        # 1. Hard Filter Stage
        st.write("### Stage 1: Hard Filter (LLM Extraction)")
        status_text = st.empty()
        status_text.info("Extracting structured constraints from prompt...")
        
        # We run the full pipeline, but we want to simulate progressive rendering
        # for a better UX, so we run it now and then render the distinct parts.
        results = run_icp_pipeline(user_prompt, hf_client, df)
        
        # Display Stage 1 Results
        status_text.empty()
        with st.expander("View Extracted JSON Constraints", expanded=True):
            st.json(results["extracted_filters"])
        
        st.metric("Surviving Companies", f"{results['survivor_count']} / {results['total_count']}")
        
        if results["error"]:
            st.error(results["error"])
            return

        with st.expander("View Surviving Companies"):
            st.write(", ".join(results["surviving_companies"]))

        st.divider()

        # 2. Semantic Search Stage
        st.write("### Stage 2: Semantic Search Ranking")
        st.success("Successfully vectorized descriptions and ranked by cosine similarity.")
        
        with st.expander("View Top Semantic Matches"):
            for i, match in enumerate(results["top_semantic_matches"]):
                st.write(f"**{i+1}. {match['company_name']}** — *Similarity: {match['similarity_score']:.3f}*")

        st.divider()

        # 3. LLM Grader Stage
        st.write("### Stage 3: RevOps Analyst ICP Scoring")
        st.success("LLM evaluation complete.")
        
        st.write("####  Final Ranked Results")
        
        for i, entry in enumerate(results["final_scores"]):
            score = entry['icp_score']
            name = entry['company_name']
            reason = entry['match_reason']
            
            # Color logic based on score
            if score >= 90:
                color = "green"
            elif score >= 80:
                color = "orange"
            else:
                color = "red"
                
            st.markdown(
                f"""
                <div style="padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 10px; background-color: #f9f9f9;">
                    <h4 style="margin: 0; color: #333;">#{i+1} {name}</h4>
                    <h2 style="margin: 5px 0; color: {color};">{score}/100</h2>
                    <p style="margin: 0; color: #555;"><i>{reason}</i></p>
                </div>
                """,
                unsafe_allow_html=True
            )
# Main UI Layout
def main():
    st.title("🎯 Hybrid ICP Scoring Engine")
    st.markdown("""
    A 3-stage pipeline that takes a natural-language Ideal Customer Profile (ICP)
    request, searches a JSON database of 100 companies, and returns a scored and
    ranked list of the best matches.
    """)

    # Sidebar for Config
    with st.sidebar:
        st.header("Configuration")
        hf_token = st.text_input("Hugging Face API Token", type="password", help="Required to run the LLM stages.")
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                st.success("Token loaded from environment!")
            else:
                st.warning("Please provide a Hugging Face token.")
        
        st.markdown("---")
        st.markdown("""
        **Pipeline Stages:**
        1. **Hard Filter:** Extracts `sector`, `employees`, `funding`, etc. using Qwen2.5-Coder-7B.
        2. **Semantic Search:** Embeds descriptions with `BAAI/bge-small-en-v1.5` and ranks top 10.
        3. **LLM Grader:** RevOps Analyst grades matches 0-100 with reasoning.
        """)

    # Main Area
    df = load_data()
    
    user_prompt = st.text_area(
        "Describe your Ideal Customer Profile (ICP):",
        height=150,
        placeholder="E.g., Series A or Series B Healthcare companies with at least 50 employees that use AI or machine learning in diagnostics or patient care. Ideally founded after 2020 and based in the United States."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Load Demo Prompt"):
            # A hack to trigger a rerun with the state updated
            st.session_state.demo_loaded = True
            st.rerun()

    if getattr(st.session_state, "demo_loaded", False):
        user_prompt = EXAMPLE_PROMPT
        st.session_state.demo_loaded = False
        st.info("Demo prompt loaded. Click 'Run Analysis' to test it.")

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        if not hf_token:
            st.error("Please provide a Hugging Face API Token in the sidebar.")
        elif not user_prompt.strip():
            st.error("Please enter an ICP description.")
        else:
            run_analysis(user_prompt, hf_token, df)


if __name__ == "__main__":
    main()
