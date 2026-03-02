"""
Hybrid ICP Scoring Engine
A 3-stage pipeline that takes a natural-language Ideal Customer Profile (ICP)
request, searches a JSON database of companies, and returns a scored and
ranked list of the best matches.

Stages:
  1. Hard Filter   — LLM extracts structured constraints → Pandas filtering
  2. Semantic Search — sentence-transformers embeddings + cosine similarity → Top 10
  3. LLM Grader     — RevOps analyst scores each company 0-100 with reasoning

Requirements:
  pip install -r requirements.txt

Usage:
  set HF_TOKEN=hf_...
  python icp_engine.py
"""

import json
import os
import sys
import time

# Force UTF-8 output on Windows to support emoji in terminal output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
load_dotenv()  # Load .env file if present

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies.json")
LLM_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10

# Example ICP prompt used when no user input is supplied
EXAMPLE_PROMPT = (
    "I'm looking for Series A or Series B Healthcare companies with at least "
    "50 employees that use AI or machine learning in diagnostics or patient care. "
    "Ideally founded after 2020 and based in the United States."
)

# Retry helper for HF Inference API calls (handles rate limits)

def _hf_call_with_retry(hf_client: InferenceClient, messages: list,
                        max_retries: int = 5) -> str:
    """Wrap hf_client.chat_completion with exponential backoff."""
    delay = 10  # seconds
    for attempt in range(1, max_retries + 1):
        try:
            response = hf_client.chat_completion(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "overloaded" in err_str or "busy" in err_str:
                if attempt == max_retries:
                    raise
                print(f"   ⏳ Rate-limited (attempt {attempt}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise


def _extract_json(text: str) -> dict | list:
    """
    Robustly extract a JSON object/array from LLM output.
    Handles cases where the LLM wraps JSON in markdown code fences.
    """
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)

# STAGE 1 — HARD FILTER (Metadata Extraction & Filtering)

def extract_hard_filters(user_prompt: str, hf_client: InferenceClient) -> dict:
    """
    Use an LLM to parse the user's natural-language ICP prompt into a
    structured dictionary of hard constraints.

    Returns a dict like:
    {
        "sector": "Healthcare" | null,
        "min_employees": 50 | null,
        "max_employees": null,
        "funding_stages": ["Series A", "Series B"] | null,
        "location_keywords": ["USA", "United States"] | null,
        "founded_after": 2020 | null,
        "founded_before": null
    }
    """
    system_msg = (
        "You are a data extraction assistant. Given a user's Ideal Customer "
        "Profile (ICP) description, extract structured hard filter constraints. "
        "Return ONLY valid JSON (no markdown fences, no explanation) with these keys:\n"
        '  "sector"           — string or null (e.g. "Healthcare", "Financial Services", '
        '"Software Development", "Manufacturing")\n'
        '  "min_employees"    — integer or null\n'
        '  "max_employees"    — integer or null\n'
        '  "funding_stages"   — list of strings or null (valid values: "Pre-seed", '
        '"Seed", "Series A", "Series B", "Series C")\n'
        '  "location_keywords"— list of location strings or null. IMPORTANT: include\n'
        '    common abbreviations/synonyms (e.g. ["USA", "United States", "US"] or\n'
        '    ["UK", "United Kingdom"]). This ensures matching against varied data formats.\n'
        '  "founded_after"    — integer year or null\n'
        '  "founded_before"   — integer year or null\n\n'
        "If the user does not mention a constraint, set it to null. "
        "Be precise — do not invent constraints that were not stated. "
        "Output ONLY the JSON object, nothing else."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    raw = _hf_call_with_retry(hf_client, messages)
    filters = _extract_json(raw)
    return filters


def apply_hard_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply the extracted hard constraints to a Pandas DataFrame of companies.
    Each filter is only applied if the value is non-null.
    Returns the filtered DataFrame.
    """
    filtered = df.copy()

    # Sector filter 
    if filters.get("sector"):
        sector = filters["sector"].strip().lower()
        filtered = filtered[filtered["sector"].str.lower() == sector]

    # Employee count range 
    if filters.get("min_employees") is not None:
        filtered = filtered[filtered["employee_count"] >= filters["min_employees"]]
    if filters.get("max_employees") is not None:
        filtered = filtered[filtered["employee_count"] <= filters["max_employees"]]

    #  Funding stage whitelist 
    if filters.get("funding_stages"):
        stages_lower = [s.strip().lower() for s in filters["funding_stages"]]
        filtered = filtered[filtered["funding_stage"].str.lower().isin(stages_lower)]

    # Location keyword matching (partial, case-insensitive) 
    if filters.get("location_keywords"):
        # Expand common country synonyms so we always match abbreviations
        synonyms = {
            "united states": ["USA", "United States", "US"],
            "usa": ["USA", "United States", "US"],
            "us": ["USA", "United States", "US"],
            "united kingdom": ["UK", "United Kingdom"],
            "uk": ["UK", "United Kingdom"],
        }
        expanded = set()
        for kw in filters["location_keywords"]:
            kw_clean = kw.strip()
            if kw_clean:
                expanded.add(kw_clean)
                for variants in synonyms.get(kw_clean.lower(), []):
                    expanded.add(variants)
        pattern = "|".join(expanded)
        if pattern:
            filtered = filtered[
                filtered["headquarters_location"].str.contains(
                    pattern, case=False, na=False
                )
            ]

    # Founding year bounds 
    if filters.get("founded_after") is not None:
        filtered = filtered[filtered["founding_year"] >= filters["founded_after"]]
    if filters.get("founded_before") is not None:
        filtered = filtered[filtered["founding_year"] <= filters["founded_before"]]

    return filtered.reset_index(drop=True)

# STAGE 2 — SEMANTIC SEARCH (Vector Embeddings & Cosine Similarity)
def semantic_search(
    filtered_df: pd.DataFrame,
    user_prompt: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Vectorize company descriptions and the user prompt using a local
    sentence-transformer model, then rank companies by cosine similarity.
    Returns the top-k companies as a DataFrame with a `similarity_score` column.
    """
    print("\n  Loading embedding model (local, no API cost)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    descriptions = filtered_df["description"].tolist()

    # Encode all descriptions + the user prompt in one batch
    desc_embeddings = model.encode(descriptions, show_progress_bar=False)
    prompt_embedding = model.encode([user_prompt], show_progress_bar=False)

    # Compute cosine similarity between the prompt and every description
    similarities = cosine_similarity(prompt_embedding, desc_embeddings)[0]

    # Attach scores and sort
    result = filtered_df.copy()
    result["similarity_score"] = similarities
    result = result.sort_values("similarity_score", ascending=False).head(top_k)

    return result.reset_index(drop=True)

# STAGE 3 — LLM GRADER (ICP Scoring by a RevOps Analyst)

def llm_grade(
    top_companies: pd.DataFrame,
    user_prompt: str,
    hf_client: InferenceClient,
) -> list[dict]:
    """
    Send the top-k company records + user's original ICP prompt to an LLM
    acting as a RevOps analyst. Returns a list of dicts:
      [
        {
          "company_name": "...",
          "icp_score": 85,
          "match_reason": "One-sentence explanation."
        },
        ...
      ]
    """
    # Build a clean JSON payload of company records for the LLM
    company_records = top_companies.drop(columns=["similarity_score"], errors="ignore")
    companies_json = company_records.to_dict(orient="records")

    system_msg = (
        "You are a senior RevOps analyst specializing in Ideal Customer Profile "
        "(ICP) scoring. You will receive:\n"
        "  1. A user's ICP description (their ideal target customer).\n"
        "  2. A list of candidate companies as JSON records.\n\n"
        "For EACH company, evaluate how well it matches the ICP and return a "
        "JSON object with the key \"results\" containing an array of objects, "
        "each with exactly these keys:\n"
        '  - "company_name": the exact company name\n'
        '  - "icp_score": an integer 0-100 (100 = perfect fit)\n'
        '  - "match_reason": strictly ONE sentence explaining the score\n\n'
        "Sort the results array from highest to lowest icp_score. "
        "Be rigorous — only give high scores (80+) to companies that closely "
        "match ALL stated criteria. "
        "Output ONLY the JSON object, nothing else."
    )

    user_msg = (
        f"### ICP Description\n{user_prompt}\n\n"
        f"### Candidate Companies\n{json.dumps(companies_json, indent=2)}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    raw = _hf_call_with_retry(hf_client, messages)
    graded = _extract_json(raw)

    # The LLM returns {"results": [...]}, extract the list
    if isinstance(graded, dict) and "results" in graded:
        return graded["results"]
    # Fallback: if the LLM returned a bare list
    if isinstance(graded, list):
        return graded
    return graded.get("results", [])

# PIPELINE ORCHESTRATOR (Importable)

def run_icp_pipeline(user_prompt: str, hf_client: InferenceClient, df: pd.DataFrame) -> dict:
    """
    Core engine logic, decoupled from CLI printing.
    Takes the prompt, client, and company database.
    Returns a dictionary with all intermediate and final states.
    """
    result = {
        "extracted_filters": {},
        "surviving_companies": [],
        "survivor_count": 0,
        "total_count": len(df),
        "top_semantic_matches": [],
        "final_scores": [],
        "error": None
    }

    # 1. Hard Filter
    filters = extract_hard_filters(user_prompt, hf_client)
    result["extracted_filters"] = filters
    
    filtered_df = apply_hard_filters(df, filters)
    result["survivor_count"] = len(filtered_df)
    
    if filtered_df.empty:
        result["error"] = "No companies survived the hard filter. Try relaxing your criteria."
        return result
        
    result["surviving_companies"] = filtered_df["company_name"].tolist()

    # 2. Semantic Search
    top_matches = semantic_search(filtered_df, user_prompt, top_k=TOP_K)
    
    # Store top matches for UI display (convert to dict list for JSON ser)
    matches_list = []
    for _, row in top_matches.iterrows():
        matches_list.append({
            "company_name": row["company_name"],
            "similarity_score": float(row["similarity_score"]),
            "description": row["description"]
        })
    result["top_semantic_matches"] = matches_list

    # 3. LLM Grader
    scored = llm_grade(top_matches, user_prompt, hf_client)
    result["final_scores"] = scored

    return result



# CLI ENTRY POINT 

def main():
    """
    End-to-end ICP scoring pipeline CLI runner.
    """
    # Validate API key 
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  Error: HF_TOKEN environment variable is not set.")
        print("   Set it with:  set HF_TOKEN=hf_...")
        sys.exit(1)

    hf_client = InferenceClient(token=hf_token)

    # Load company data 
    print("  Loading company data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        companies = json.load(f)
    df = pd.DataFrame(companies)
    print(f"   Loaded {len(df)} companies.\n")

    # Get ICP prompt 
    print("=" * 70)
    print("  HYBRID ICP SCORING ENGINE")
    print("=" * 70)
    user_input = input(
        "\nDescribe your Ideal Customer Profile (or press Enter for demo):\n> "
    ).strip()
    user_prompt = user_input if user_input else EXAMPLE_PROMPT
    print(f"\n  ICP Prompt:\n   \"{user_prompt}\"\n")

    # RUN PIPELINE
    results = run_icp_pipeline(user_prompt, hf_client, df)

    print("─" * 70)
    print("  STAGE 1 → Hard Filter (LLM Metadata Extraction)")
    print("─" * 70)
    print(f"\n  Extracted Constraints:\n{json.dumps(results['extracted_filters'], indent=4)}")
    print(f"\n  Companies after hard filter: {results['survivor_count']} / {results['total_count']}")
    
    if results["error"]:
        print(f"\n  {results['error']}")
        sys.exit(0)

    print("   Survivors:", ", ".join(results["surviving_companies"]))

    print(f"\n{'─' * 70}")
    print("  STAGE 2 → Semantic Search (Embedding Similarity)")
    print("─" * 70)
    print(f"\n  Top {len(results['top_semantic_matches'])} Semantic Matches:")
    for i, match in enumerate(results['top_semantic_matches']):
        print(f"   {i + 1:>2}. {match['company_name']:<30} (similarity: {match['similarity_score']:.4f})")

    print(f"\n{'─' * 70}")
    print("  STAGE 3 → LLM Grader (RevOps Analyst Scoring)")
    print("─" * 70)

    print(f"\n{'═' * 70}")
    print("  FINAL ICP SCORES")
    print("═" * 70)
    for i, entry in enumerate(results["final_scores"]):
        print(
            f"\n  #{i + 1}  {entry['company_name']}"
            f"\n      ICP Score   : {entry['icp_score']}/100"
            f"\n      Reason      : {entry['match_reason']}"
        )

    print(f"\n{'═' * 70}")
    print(f"  Pipeline complete — {len(results['final_scores'])} companies scored.")
    print("═" * 70)

    return results["final_scores"]


if __name__ == "__main__":
    main()
