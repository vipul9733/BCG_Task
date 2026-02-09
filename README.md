# AI Research Paper Summarizer (Evidence-Grounded)

Deployed app: **<PASTE_YOUR_STREAMLIT_URL_HERE>**

## What problem does this solve?
Researchers, students, and consultants often spend hours reading papers to:
- extract key findings and methodology
- compare multiple papers
- identify contradictions and gaps
- answer questions with citations

This app accelerates that workflow by generating **structured summaries**, **cross-paper synthesis**, and **evidence-grounded Q&A**.

## Key AI features (AI is essential)
1. **Structured extraction with page citations**
   - Extracts research question, methodology, findings, limitations, and contributions.
2. **Cross-paper synthesis**
   - Finds common themes, contradictions, gaps, and synthesizes the set.
3. **Evidence-grounded Q&A**
   - Retrieves relevant chunks (local TF-IDF) and answers using only retrieved evidence with citations.
4. **Reference overlap network**
   - Builds a similarity network based on shared normalized references (heuristic co-citation).

## How it works (high level)
- PDFs → text extracted per page
- Pages → overlapping chunks with page numbers
- Retrieval: local TF-IDF ranks chunks for a user question (no LLM call for retrieval)
- LLM (Claude) generates:
  - paper-level structured extraction
  - cross-paper synthesis
  - Q&A answer constrained to provided evidence

## Run locally
### 1) Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
