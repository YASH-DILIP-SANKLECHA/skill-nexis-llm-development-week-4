# Assignment 4 — AI Assistant Evaluation

`evaluation_test_set.csv` contains 20 questions: 10 direct, 5 multi-passage, 5 absent-answer questions.

From the project root, run:
```powershell
python evaluate.py
```
This creates `evaluation_results.csv` and `evaluation_report.md`. Review the answer and source manually, then fill `answer_correctness_human_review` and `source_accuracy_human_review`. The script does not fabricate human judgement.
