For **TraceX**, I recommend dividing the work among three people based on independent technical contributions, with clearly defined ownership so that everyone has substantial implementation work and can explain their contribution during the viva.

# **Project division – 3 members**

**Member 1 – Data Engineering & Preprocessing**

Data pipeline

**Primary responsibility:** Prepare reliable datasets and establish the experimental protocol.

Tasks:

- Download Foursquare NYC, TKY and Massive-STEPS.
- Clean missing values, duplicates and invalid records.
- Extract user IDs, POI IDs, timestamps and categories.
- Sort check-ins chronologically.
- Divide trajectories using the 24-hour session gap.
- Create training, validation and test splits.
- Ensure no target leakage.
- Analyse dataset statistics and POI coverage.

**Deliverables:** Clean datasets, preprocessing scripts, dataset statistics and leakage checks.

**Member 2 – Information Retrieval & Baselines**

Core algorithms

**Primary responsibility:** Implement retrieval techniques and the non-LLM baseline.

Tasks:

- Implement frequency-based POI prediction.
- Implement Jaccard similarity for historical sequence selection.
- Implement Longest Common Subsequence (LCS).
- Implement TF-IDF and cosine similarity.
- Implement BM25 retrieval.
- Experiment with hybrid retrieval if time permits.
- Compare retrieval quality using Precision\@K, Recall\@K, MAP and MRR.

**Deliverables:** Retrieval modules, baseline predictions, retrieval evaluation and comparison graphs.

**Member 3 – LLM Integration, Experiments & Interface**

AI & evaluation

**Primary responsibility:** Integrate the language model and build the experimental demonstration.

Tasks:

- Set up Qwen2.5-7B-Instruct (4-bit quantisation).
- Implement the LLM-Mob-style prediction prompt.
- Integrate retrieved demonstrations from Member 2.
- Generate and validate ranked POI predictions.
- Conduct the revisit-versus-new experiment.
- Conduct the venue-name removal and aliasing experiment on Massive-STEPS.
- Calculate Accuracy\@1, Accuracy\@5, MRR and NDCG\@5.
- Build the Streamlit demonstration dashboard.

**Deliverables:** LLM pipeline, experiment results, evaluation visualisations and working demo.

## **Shared responsibilities**

All three members should contribute to these tasks:

| Work                   | Responsibility                                    |
| ---------------------- | ------------------------------------------------- |
| Literature review      | Each member finds 2–3 relevant papers             |
| Research methodology   | Discuss and finalise together                     |
| Results interpretation | All three                                         |
| Research paper/report  | Each writes their own methodology/results section |
| GitHub integration     | All three review code                             |
| Viva preparation       | Everyone understands the complete pipeline        |

## **How the three contributions connect**

**Member 1: Dataset preparation**

Cleaning · Sessionisation · Train/test split

**Member 2**

IR algorithms · Frequency baseline · Retrieval

**Member 3**

Qwen · Prediction · Experiments · Demo

**Integrated TraceX System**

Comparative evaluation · Results · Research report

## **Suggested contribution distribution**

| Member   | Approx. contribution |
| -------- | -------------------- |
| Member 1 | 30%                  |
| Member 2 | 35%                  |
| Member 3 | 35%                  |

These are suggested workload shares, not fixed grading percentages.

**One important recommendation:** Member 2 and Member 3 should coordinate closely. The retrieval methods must return demonstrations in exactly the format expected by the LLM. Otherwise, integration becomes unnecessarily difficult.

Also, keep the original three experiments in your brief as the project's core: baseline comparison (E1), revisit versus new POIs (E2), and venue-name disguise (E3). Don't let optional IR extensions delay these experiments.&#x20;

Copy, Recall, or Reason Project Brief (1).pdf



My suggestion is to create three GitHub branches — `data-pipeline`, `retrieval-engine`, and `llm-experiments` — and merge them into `main` once each module passes testing.