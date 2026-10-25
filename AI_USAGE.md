# AI Usage Disclosure

**Project:** GradCFA Reproduction and Extension (DA351 Explainable AI)

**Tool used:** Claude (Anthropic), through the claude.ai chat interface.

## What I used it for

- Choosing and checking a project paper (venue ranking, dataset size, code availability)
- Explaining the GradCFA paper section by section
- Planning the work and the timeline
- Debugging errors that appeared when running the original repository
- Drafting a message to the course professor

## Prompts used (summary)

- Criteria for comparing candidate research papers
- "Verify that this GitHub repository is the official code for the paper"
- Error messages pasted from my terminal and notebook, with the question "what should I do"
- "Explain this paper section by section"
- "Give me a plan for the remaining days"

## How I verified and modified the output

- I ran every code change myself and checked that the error was resolved before moving on.
- Fixes to the original repository are listed in `README.md` under "Issues found in the original repository".
- Numbers from my runs are compared against the paper's own tables, which I read in the paper.
- Suggestions that I could not verify by running them are not used as results.

## Log

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

Add one row for every new session.