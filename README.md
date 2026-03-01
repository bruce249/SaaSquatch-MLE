# Hybrid ICP Scoring Engine

The Hybrid Ideal Customer Profile (ICP) Scoring Engine is a 3-stage modular pipeline designed to match natural-language customer requirements against a structured database of companies. It leverages Large Language Models (LLMs) for constraint extraction and reasoning, paired with local semantic search for accurate similarity ranking.

## System Architecture

The pipeline processes a natural-language prompt through three distinct stages:

### Stage 1: Hard Filter (LLM Metadata Extraction)
The system uses an LLM (`Qwen/Qwen2.5-Coder-7B-Instruct`) to extract hard constraints from the user's prompt. It parses the request into a structured JSON object containing:
- Sector
- Employee Count (Min/Max)
- Funding Stages (e.g., Series A, Series B)
- Location Keywords
- Founding Year bounds

These constraints are then applied via Pandas dataframe filtering to immediately eliminate companies that do not meet the strict criteria. The system includes synonym expansion specifically for United States location matching to ensure robustness against varied data formats.

### Stage 2: Semantic Search (Vector Similarity)
The companies that survive the hard filter are passed to a local embedding model (`BAAI/bge-small-en-v1.5`). The system vectorizes both the company descriptions and the original user ICP prompt. It calculates the cosine similarity between the prompt vector and each company vector, ranking the results and selecting the Top 10 semantic matches. Using a local embedding model avoids latency and API costs for bulk similarity comparisons.

### Stage 3: LLM Grader (RevOps Analyst Scoring)
The Top 10 semantic matches are sent to the LLM. The LLM acts as a senior RevOps analyst, evaluating each company against the user's detailed ICP description. It returns a scored JSON array where each company receives:
- An ICP Score between 0 and 100
- A precise, one-sentence explanation justifying the score based on the matching criteria

## Project Structure

- `app.py`: The Streamlit web application providing the graphical user interface.
- `icp_engine.py`: The core pipeline orchestration and command-line interface logic.
- `companies.json`: A standard JSON database containing records of 100 sample companies.
- `requirements.txt`: Python package dependencies.

## Setup and Installation

### Prerequisites
- Python 3.9 or higher is recommended.
- A Hugging Face account and API Token (with access to the Inference API).

### Installation Steps

1. Clone or navigate to the project repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Hugging Face API Token in your environment:
   ```bash
   # Windows (PowerShell)
   $env:HF_TOKEN="your_huggingface_token"
   
   # Linux/macOS
   export HF_TOKEN="your_huggingface_token"
   ```

## Usage

You can run the engine either via the interactive Web UI or as a Command-Line Interface.

### Option A: Web Interface (Streamlit)
The recommended way to interact with the engine. It offers a clean, progressive loading interface with dynamic metric cards.

```bash
streamlit run app.py
```
This will launch the application locally at `http://localhost:8501`. If you have not set the environment variable, you can securely input your Hugging Face token directly in the application's sidebar.

### Option B: Command-Line Interface (CLI)
The original execution method, useful for terminal environments or automated scripts.

```bash
python icp_engine.py
```
Upon running, you will be prompted to enter your Ideal Customer Profile description. Pressing Enter without typing will load a default demonstration prompt.

## Configuration

The core models and operational parameters can be adjusted directly in the configuration section of `icp_engine.py`:

- `LLM_MODEL`: Defines the Hugging Face model used for Stage 1 extraction and Stage 3 grading (currently set to `Qwen/Qwen2.5-Coder-7B-Instruct`).
- `EMBEDDING_MODEL_NAME`: Defines the local sentence-transformer model used for Stage 2 (currently set to `BAAI/bge-small-en-v1.5`).
- `TOP_K`: Determines how many companies proceed from Stage 2 to Stage 3 grading.
- `DATA_FILE`: Points to the local JSON database file.
