# Week 4 AI Assistant Evaluation Report

## Test design
- Total questions: 20
- Directly answered: 10 (required 10)
- Multi-passage: 5 (required 5)
- Answers absent: 5 (required 5)

## Automated measurements
- Mean retrieval/response time: 1.54 ms
- Absent-question abstention checks passed: 2/5
- Full per-question results: `evaluation_results.csv`

## Manual review required
Fill `answer_correctness_human_review` and `source_accuracy_human_review` after checking each answer against its retrieved evidence. Similarity scores are not correctness scores. For multi-passage questions, check that all necessary passages were retrieved.

## Limitations
This reproducible baseline uses TF-IDF retrieval and extractive evidence responses rather than a generative LLM. It requires no paid API. A production version should add a local/hosted generator, measure groundedness, and calibrate abstention thresholds.
