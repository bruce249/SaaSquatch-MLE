# Hybrid ICP Scoring Engine

Welcome to the **Hybrid ICP Scoring Engine**! 

If you are new to sales, marketing, or programming, don't worry-this guide is written so that anyone can understand what this tool does, why it is useful, and exactly how to run it on your own computer.

---

## What is this tool? 

In business, **ICP** stands for **"Ideal Customer Profile."** It describes the perfect type of company you want to sell your product or service to. For example, your ICP might be: *"Healthcare companies with more than 50 employees, located in the United States, that use Artificial Intelligence."*

Usually, finding companies that match your ICP means a human has to manually read through hundreds of company descriptions and guess which ones are the best fit.

**This tool automates that entirely using AI.** 

You simply type out your Ideal Customer Profile in plain English. The engine will read a database of companies and use Artificial Intelligence to filter, rank, and score every single company, returning a list of the absolute best matches.

---

## How does it work? (The 3 Stages)

When you hit "Run," the engine processes your request in three steps:

### Stage 1: The Hard Filter
The system uses an advanced AI (like ChatGPT, but using an open-source model called `Qwen2.5-Coder-7B`) to read your prompt and extract the "hard facts." 
* Did you mention a specific location? Employee constraints? A certain industry?
The AI pulls these out and instantly deletes any companies from the database that don't satisfy these basic rules.

### Stage 2: The Semantic Search
Next, the system reads the descriptions of the surviving companies. Instead of just looking for exact keyword matches (which is clumsy), it uses an AI "Embedding Model" (`BAAI/bge-small-en-v1.5`). This model understands the *meaning* of words. It compares the meaning of your ICP to the meaning of each company's description and ranks the Top 10 closest matches.

### Stage 3: The AI Evaluator (RevOps Analyst)
Finally, the system sends those Top 10 companies back to the main AI. It asks the AI to act like a strict business analyst. The AI reads each company's profile deeply, gives it a score out of 100 based on how well it matches your original prompt, and provides a one-sentence explanation for *why* it gave that score.

---

## Step-by-Step Installation Guide

Ready to try it yourself? Follow these steps exactly.

### Step 1: Getting Ready
1. You need **Python** installed on your computer. (If you don't have it, download it from python.org).
2. You need a free **Hugging Face** account. Hugging Face hosts the AI models we use. 
   - Go to huggingface.co and create an account.
   - Go to your account Settings -> "Access Tokens" and create a new token. Copy this token (it usually starts with `hf_`). Keep it secret!

### Step 2: Downloading the Code
If you are reading this on GitHub, click the green "Code" button and select "Download ZIP". Extract the folder to your computer.

### Step 3: Installing Dependencies
Open your computer's Terminal (Mac) or Command Prompt / PowerShell (Windows). Navigate to the folder where you extracted the code.

Type the following command and press Enter:
```bash
pip install -r requirements.txt
```
*This downloads the necessary extra code packages required to run the AI, data tools, and the visual website interface.*

---

## Running the Application

This project comes with a beautiful, easy-to-use Web Interface.

In your terminal, while inside the project folder, type:
```bash
streamlit run app.py
```

### Using the Web Interface
1. Once you run the command above, a window will automatically open in your web browser.
2. Looking at the **sidebar on the left**, paste your Hugging Face Access Token (the one starting with `hf_`) into the password box.
3. In the **main text box**, type out what kind of companies you are looking for. (Or just click the "Load Demo Prompt" button for an example).
4. Click the large **"Run Analysis"** button.
5. Sit back and watch the AI work! It will show you the exact constraints it pulled out, the number of companies remaining, and eventually generate beautiful, color-coded scorecards for your top 10 leads.

---

## What are these files?
For the curious, here is what the files in this folder actually do:
- `app.py`: This contains the code that draws the buttons, text boxes, and Web Interface you see in your browser.
- `icp_engine.py`: This is the real "brain" of the operation. It contains the code that talks to the AI, filters the data, and does the mathematical ranking.
- `companies.json`: This is a sample database of 100 fake companies used for testing the engine. You can eventually replace this with your own sales database!
- `requirements.txt`: A simple text file that tells your computer what extra Python packages to install.

---

*Enjoy finding your perfect sales leads effortlessly!*
