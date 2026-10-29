"""Run 20-question evaluation. Human review columns remain blank intentionally."""
import csv, time
from pathlib import Path
from rag_assistant import read_upload, KnowledgeBase, answer_from_evidence, ABSTAIN
ROOT=Path(__file__).resolve().parent
class File:
    def __init__(self,p): self.name=str(p)
passages=[]
for p in sorted((ROOT/"sample_documents").glob("*.pdf")): passages += read_upload(File(p))
kb=KnowledgeBase(passages)
test=ROOT/"Assignment_4_AI_Assistant_Evaluation"/"evaluation_test_set.csv"
out=ROOT/"Assignment_4_AI_Assistant_Evaluation"/"evaluation_results.csv"
report=ROOT/"Assignment_4_AI_Assistant_Evaluation"/"evaluation_report.md"
results=[]
for row in csv.DictReader(test.open(encoding="utf-8")):
    start=time.perf_counter(); hits=kb.retrieve(row["question"],4); ans=answer_from_evidence(row["question"],hits)
    elapsed=(time.perf_counter()-start)*1000
    sources="; ".join(f"{p.document}:p{p.page} (sim={s:.3f})" for p,s in hits)
    abstain=("PASS" if ans==ABSTAIN else "FAIL - review") if row["category"]=="absent" else "N/A"
    results.append({**row,"retrieved_sources":sources,"top_similarity":f"{hits[0][1]:.3f}" if hits else "0.000","answer_or_abstention":ans,"response_time_ms":f"{elapsed:.2f}","abstention_check":abstain,"answer_correctness_human_review":"","source_accuracy_human_review":"","review_notes":""})
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
counts={c:sum(r["category"]==c for r in results) for c in ("direct","multi_passage","absent")}
mean=sum(float(r["response_time_ms"]) for r in results)/len(results)
passed=sum(r["abstention_check"]=="PASS" for r in results if r["category"]=="absent")
report.write_text(f"""# Week 4 AI Assistant Evaluation Report

## Test design
- Total questions: {len(results)}
- Directly answered: {counts['direct']} (required 10)
- Multi-passage: {counts['multi_passage']} (required 5)
- Answers absent: {counts['absent']} (required 5)

## Automated measurements
- Mean retrieval/response time: {mean:.2f} ms
- Absent-question abstention checks passed: {passed}/{counts['absent']}
- Full per-question results: `evaluation_results.csv`

## Manual review required
Fill `answer_correctness_human_review` and `source_accuracy_human_review` after checking each answer against its retrieved evidence. Similarity scores are not correctness scores. For multi-passage questions, check that all necessary passages were retrieved.

## Limitations
This reproducible baseline uses TF-IDF retrieval and extractive evidence responses rather than a generative LLM. It requires no paid API. A production version should add a local/hosted generator, measure groundedness, and calibrate abstention thresholds.
""",encoding="utf-8")
print(f"Evaluated {len(results)} questions. Mean latency: {mean:.2f} ms")
print(f"Results: {out}")
print(f"Report: {report}")
