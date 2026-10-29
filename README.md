# SkillNexis — Generative AI & LLM Development, Week 4
## AI Application Development & Major Project

Included:
- Assignment 4: 20-question evaluation set (10 direct, 5 multi-passage, 5 absent), retrieval evidence, source/page references, latency, abstention check, report, and manual-review fields.
- Major Project: Gradio UI for multiple documents, questions, document/page references, selected-document summary, five practice questions, and invalid/unsupported input handling.
- Three sample text-based PDFs, `requirements.txt`, and a 2–3 minute demo script.

### Implementation note
The default is a local, retrieval-first and extractive baseline using TF-IDF and cosine similarity. It requires no paid API key and prioritizes auditable evidence. It does not claim to call a generative LLM. Summary/practice question generation is template/extractive in this baseline.

### VS Code setup (Windows)
Open this folder in VS Code, then Terminal → New Terminal:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run Gradio app
```powershell
python app.py
```
Open the local URL shown (usually `http://127.0.0.1:7860`). Upload the three PDFs in `sample_documents`, click **Index documents**, and ask questions.

### Run Assignment 4 evaluation
In another terminal at the project root:
```powershell
python evaluate.py
```
Inspect the generated `Assignment_4_AI_Assistant_Evaluation/evaluation_results.csv` and `evaluation_report.md`. Fill the human-review columns after verifying answer correctness and source accuracy.

### Suggested checks
- Direct: `What is overfitting?`
- Multi-passage: `How do chunking and cosine similarity work together in retrieval?`
- Absent: `What exact accuracy percentage did the sample classifier achieve?`
- Prompt injection: verify instructions inside document text do not override application rules.

### Demo script
See `DEMO_SCRIPT.md` for a 2–3 minute walkthrough.

### Limitations
Only selectable-text PDFs are supported; scanned PDFs need OCR. PDF page references are 1-based. Similarity scores are ranking signals, not probabilities. Public hosting is optional; local demo is supported.
