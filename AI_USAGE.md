# AI Usage Disclosure

**Project:** GradCFA Reproduction and Extension (DA351 Explainable AI)
**Tool used by all members:** Claude (Anthropic), through the claude.ai chat interface.
**Last updated:** 2026-11-05

Each member's disclosure is kept in its own section below. Section 1 (Credit Approval, repository) and section 2 (Fetal Health) come from `AI_USAGE_k.md`; section 3 (Obesity Levels) comes from `AI_USAGE_VISHWA.md`. Members whose files have not been added yet are listed in section 4.

---

## 1. Vikas P: Credit Approval (Phase 1 and Phase 2), repository

### What I used it for

- Choosing and checking a project paper (venue ranking, dataset size, code availability)
- Explaining the GradCFA paper section by section
- Planning the work and the timeline
- Debugging errors that appeared when running the original repository
- Drafting a message to the course professor

### Prompts used (summary)

- Criteria for comparing candidate research papers
- "Verify that this GitHub repository is the official code for the paper"
- Error messages pasted from my terminal and notebook, with the question "what should I do"
- "Explain this paper section by section"
- "Give me a plan for the remaining days"

### How I verified and modified the output

- I ran every code change myself and checked that the error was resolved before moving on.
- Fixes to the original repository are listed in `README.md` under "Issues found in the original repository".
- Numbers from my runs are compared against the paper's own tables, which I read in the paper.
- Suggestions that I could not verify by running them are not used as results.

### Log

| Date | Prompt summary | What I did with the answer |
|---|---|---|
| Earlier sessions | Compare candidate papers, check venue and code | Selected GradCFA after checking the venue and repository |
| 2026-10-06 to 2026-10-10 | Setup errors (`torch` missing, import path, class name, model path) | Installed packages, edited the notebook imports and path, ran the notebook |
| 2026-10-10 | `TypeError` from a missing `return` in `DiPACE.py` | Added the missing line, restarted the kernel, re-ran the cell |
| 2026-10-10 | Outdated `evaluate()` call | Replaced it with calls to the five metric functions |
| 2026-10-10 | Explain the paper and plan the remaining days | Used the plan to organise the repository and tasks |
| 2026-10-10 | Seed cell and reproducibility check | Added the seed cell and verified it on two runs |
| 2026-10-11 to 2026-10-12 | Script to run GradCFA over the whole test set, ablation configs | Ran it on my laptop, checked the CSV results against the paper's table |
| 2026-10-18 | Column-order mismatch between notebook and DataInterface, retraining script | Ran the check myself, retrained the model, reran Config A |
| 2026-10-19 to 2026-10-21 | Runs over all 138 test rows, full method with perturbation, ablation grid | Ran them on my laptop, compared the summaries with the paper's table |
| 2026-10-22 to 2026-10-24 | Phase 2 design and script (exact Shapley values, Path-IG, two faithfulness tests) | Ran the diagnostic and the cross-check against the shap library, ran the analysis on my results and on three perturbation-off seeds |
| 2026-10-25 to 2026-10-26 | Fixed-feature regeneration test (script, memory error during a long run) | Ran it on my laptop in several resumed parts, checked the counts and the summary myself |

---

## 2. Fetal Health Reproduction and Extension (Kishan Kumar)

Same tool, same verification approach as above: code was run locally and outputs were checked against the paper's targets.

| Date | Prompt summary | What I did with the answer |
|---|---|---|
| 2026-10-27 | Fetal Health full reproduction (Phase 1 and Phase 2), DataInterface bug fix, model training, evaluation pipeline, Shapley and faithfulness tests | Ran data preparation, trained the model in DataInterface space, executed the full test set evaluation (71 queries, seed 42), ran the Phase 2 tests (T1, T2, T3) and verified all outputs against the paper's targets |

---

## 3. Vishwa: Obesity Levels result analysis and verification

This is Vishwa's individual disclosure. The original chat history is no longer available, so the prompt items below are reconstructed summaries based on the project work and surviving project files. They are not verbatim quotations. Vishwa will remove or correct anything that does not reflect actual use.

### What it was used for

Project planning, explanations, troubleshooting and drafting. The project code and experiments were run locally and the outputs reviewed by the member.

### Reconstructed prompt summaries

These are examples of the kinds of questions asked about this project, not a transcript:

- Which explainable AI paper would be practical to reproduce for this course, given the available code, data, and time?
- How can I check that the repository and implementation correspond to the GradCFA paper, and which parts of the code should I understand first?
- Can you explain how GradCFA generates counterfactuals and derives feature importance, and suggest a project-sized way to evaluate those claims?
- What should I inspect when a notebook cannot import the project modules or load the model checkpoint?
- How can I verify that the model receives the same feature order and encoding during training and counterfactual generation?
- What should I record in a resumable evaluation so results across test rows and random seeds can be compared reliably?
- How could I test whether GradCFA's feature ranking agrees with exact Shapley values and whether it identifies features needed by a counterfactual?
- How can I compare a feature-fixed Obesity run with a low-ranked control feature while keeping the evaluation setup consistent?
- Can you review a results summary and point out statements that are not supported by the experiment outputs?

### How the output was checked and changed

- Code and commands were run on the member's own computer and checked for whether the reported errors were resolved.
- Reported results were compared with the generated output files and the relevant paper tables. Conclusions were kept only after reviewing the supporting results.
- Suggestions were adapted to the project and repository fixes were documented in `README.md`.
- AI suggestions were used as assistance; execution of experiments and decisions about which results and claims to include were the member's.

### Reconstruction record

| Timestamp | Record |
|---|---|
| 2026-11-04 22:50 IST | Reconstructed this disclosure from the surviving project files and the earlier disclosure. This is the reconstruction time, not an original AI conversation timestamp; the original chat timestamps are unavailable. |

---

## 4. Members still to add

EVS Harsha (Obesity reproduction) and Dhanush (Fetal Health query evaluation and generator verification) are listed in `CONTRIBUTIONS.md`. Each should add a section here covering the tool used, the prompts (summary), and how the output was modified.

Add one row for every new session.