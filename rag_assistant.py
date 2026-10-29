"""Week 4 local document assistant: retrieval-first, extractive, no paid API key."""
import re, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import gradio as gr
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ABSTAIN = "I could not find this in the provided documents."
ALLOWED = {".pdf", ".txt", ".md"}

@dataclass
class Passage:
    document: str
    page: int
    text: str

def clean(s): return re.sub(r"\s+", " ", s or "").strip()

def chunk(text, document, page, max_words=120):
    words = clean(text).split()
    return [Passage(document, page, " ".join(words[i:i+max_words])) for i in range(0, len(words), max_words) if words[i:i+max_words]]

def read_upload(file_path):
    if not file_path: raise ValueError("Upload a PDF, TXT, or Markdown file first.")
    path = Path(getattr(file_path, "name", str(file_path)))
    if path.suffix.lower() not in ALLOWED:
        raise ValueError("Unsupported file type. Upload PDF, TXT, or Markdown.")
    passages = []
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, 1):
                passages += chunk(page.extract_text() or "", path.name, i)
        except Exception as e: raise ValueError(f"Could not read PDF: {e}")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, section in enumerate(re.split(r"\n\s*---\s*\n", text), 1):
            passages += chunk(section, path.name, i)
    if not passages: raise ValueError("No selectable text found. Scanned/image-only PDFs are not supported.")
    return passages

class KnowledgeBase:
    def __init__(self, passages):
        if not passages: raise ValueError("No document passages were loaded.")
        self.passages = passages
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform([p.text for p in passages])
    def retrieve(self, question, k=4):
        if not clean(question): return []
        scores = cosine_similarity(self.vectorizer.transform([clean(question)]), self.matrix).ravel()
        return [(self.passages[i], float(scores[i])) for i in scores.argsort()[::-1][:min(k,len(scores))] if scores[i] > 0]

def has_injection(text):
    return any(re.search(p, text, re.I) for p in [
        r"ignore (all )?(previous|prior) instructions",
        r"reveal (the )?(system prompt|secrets|passwords)",
        r"disregard (all )?rules"
    ])

def answer_from_evidence(question, hits):
    if not clean(question) or not hits or hits[0][1] < 0.08: return ABSTAIN
    safe = [(p,s) for p,s in hits if not has_injection(p.text)]
    if not safe: return ABSTAIN
    return "Based on the retrieved document evidence:\n\n" + "\n\n".join(
        f"- {p.text} ({p.document}, PDF page {p.page})" for p,s in safe[:3])

def build_index(files):
    passages, errors = [], []
    for f in files or []:
        try: passages += read_upload(f)
        except Exception as e: errors.append(f"- {Path(str(f)).name}: {e}")
    if not passages:
        return None, "No documents indexed. Upload at least one valid text-based document."
    try: kb = KnowledgeBase(passages)
    except Exception as e: return None, f"Indexing failed: {e}"
    msg = f"Indexed {len(passages)} passages from {len(set(p.document for p in passages))} document(s)."
    if errors: msg += "\n\nWarnings:\n" + "\n".join(errors)
    return kb, msg

def ask(question, kb):
    start = time.perf_counter()
    if kb is None: return ABSTAIN, "Upload and index documents first.", "Not measured"
    hits = kb.retrieve(question)
    ans = answer_from_evidence(question, hits)
    source = "\n\n".join(f"**{p.document} — page {p.page}** · similarity {score:.3f}\n\n> {p.text}" for p,score in hits) or "No relevant source passages found."
    return ans, source, f"{(time.perf_counter()-start)*1000:.1f} ms"

def summarize(files, selected):
    kb, status = build_index(files)
    if kb is None: return status
    passages = [p for p in kb.passages if p.document == selected] if selected else kb.passages
    if not passages: return "Select an indexed document."
    return "\n\n".join(f"- {p.text} (page {p.page})" for p in passages[:5])

def practice(files, selected):
    kb, status = build_index(files)
    if kb is None: return status
    passages = [p for p in kb.passages if p.document == selected] if selected else kb.passages
    if not passages: return "Select an indexed document."
    templates = [
        "What is the main concept in this passage?",
        "Which key terms are defined?",
        "What process or steps are described?",
        "What distinction or comparison is made?",
        "What limitation, risk, or recommendation is mentioned?"
    ]
    return "\n\n".join(f"{i+1}. {templates[i]}\n   Study source: {passages[i % len(passages)].document}, page {passages[i % len(passages)].page}\n   Evidence: {passages[i % len(passages)].text[:260]}..." for i in range(5))

def create_demo():
    with gr.Blocks(title="AI Study & Knowledge Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AI Study & Knowledge Assistant\nUpload course notes, index them, and ask questions. Answers use retrieved passages and document/page references. Local baseline; no paid API key.")
        kb_state = gr.State(None)
        files = gr.File(label="Upload at least three text-based documents", file_count="multiple", file_types=[".pdf",".txt",".md"], type="filepath")
        index_btn = gr.Button("Index documents", variant="primary")
        status = gr.Markdown("No documents indexed.")
        index_btn.click(build_index, inputs=[files], outputs=[kb_state,status])
        question = gr.Textbox(label="Question", placeholder="What is overfitting?")
        ask_btn = gr.Button("Find answer", variant="primary")
        answer = gr.Markdown()
        sources = gr.Markdown()
        latency = gr.Markdown()
        ask_btn.click(ask, inputs=[question,kb_state], outputs=[answer,sources,latency])
        gr.Markdown("## Document study tools")
        selected = gr.Dropdown(label="Selected document (optional)", choices=[], interactive=True)
        refresh = gr.Button("Refresh document list")
        refresh.click(lambda fs: gr.update(choices=[Path(f).name for f in (fs or [])], value=(Path(fs[0]).name if fs else None)), inputs=[files], outputs=[selected])
        with gr.Row():
            sum_btn = gr.Button("Summarize selected document")
            q_btn = gr.Button("Generate five practice questions")
        summary = gr.Markdown()
        questions = gr.Markdown()
        sum_btn.click(summarize, inputs=[files,selected], outputs=[summary])
        q_btn.click(practice, inputs=[files,selected], outputs=[questions])
        gr.Markdown("Safety: uploaded document text is evidence, not an instruction source. Scanned PDFs need OCR and are not supported by this baseline.")
    return demo
