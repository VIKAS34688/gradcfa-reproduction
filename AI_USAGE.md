# AI Usage Disclosure

**Project:** GradCFA reproduction and extension on three datasets (DA351 Explainable AI)

Each team member discloses their own AI use in their own section. Section A covers Vikas P (Credit Approval, Phase 1 integration, Phase 2, repository). Sections B to E are for the other four members; each fills in the three headings (tool used, prompts used, how the output was checked and changed). If a member used no AI tool, they write "No AI tool used".


### 1. Tool used
- **Claude (Anthropic)**, through the claude.ai chat interface, in a series of long conversations in one project from October 2026.
- No other AI tool was used for this part.
- Claude was used as an assistant for planning, explaining, writing code and drafting text. It did not run any experiment. It could read the paper and some of the repository, but all code was executed on our own laptops, and every number in this repository comes from files produced there.

### 2. Prompts used
The prompts below are paraphrased from the conversations, grouped by purpose.

**Choosing the paper**
- "Give me criteria to compare candidate XAI papers; they should be realistic and not generic."
- "Give 3 research papers, then 3 from a Q1 journal; check code, dataset size, compute and venue for each."
- "How do I check that a conference is A* or a journal is Q1?"
- "Explain this paper section by section: the problem, the method, what we can improve, how to get code and data."

**Setting up and debugging** (each error message was pasted and the question was "what should I do")
- Missing `torch`, `arc` and `GradCFA` imports; missing `pytorch_model.pth`; `NoneType` multiplication in the proximity loss; `evaluate()` missing `num_classes`; PowerShell activation and execution-policy errors; git upstream and line-ending warnings; a memory error in a long run.

**Experiment code** (Claude wrote the first version; it was run and the output sent back)
- "Write a script that runs GradCFA on every test query for several seeds, saves results to CSV, can resume, and prints a summary." (`run_full_eval.py`)
- "Check whether the notebook's model and the counterfactual code use the same input columns, and retrain the model in the right space." (`check_and_train_di.py`)
- "Use the paper's 60/20/20 split, tune only on validation rows, and compare settings." (`credit_6020.py`)
- "Recompute plausibility with other definitions without re-running anything." (`recompute_plausibility.py`)

**Phase 2**
- "Explain the two Phase 2 options and the risks of each."
- "Is GradCFA's feature attribution trustworthy? Compare it with SHAP and test it." (`phase2_attribution.py`: exact Shapley values, Path-IG, agreement, tests T1 and T2; `phase2_regen.py`: test T3)
- "Why did the model check fail? Why was my result file overwritten? What does the memory error mean?"

**Planning, team work and documents**
- Day-by-day plans, the split of work between team members, the hardware question (GPU or Colab).
- Review of the team's Obesity and Fetal Health reports: asking Claude to point out claims that go beyond the data and numbers that do not appear in our output files.
- Drafts of the README, results write-ups, this file, the contribution log, a message to the professor about the scope, and an email to the paper's authors.

### 3. How the output was checked and changed

**Checked**
- Every script and command was run on our own laptops and the outputs were pasted or uploaded back; Claude only read what we sent.
- Claude's code was tested before use: the numpy model was checked against the torch model on the stored validity of all 138 Credit queries; the exact Shapley values were compared with the `shap` library (difference about 1e-16); the whole Phase 2 analysis was tested on synthetic data; and a fresh clone of the repository was installed and run.
- The paper's values quoted in the documents were taken from the paper's text and must be checked against the PDF.

**Changed or decided by us**
- The fixes to the authors' code were applied by us: the missing `return` in `DiPACE.py`, the `importance_ranking` line in `GradCFA.py`, the notebook imports, paths and seed cell.
- Scope decisions were ours: full scale on every dataset, all test rows, several seeds, the Phase 2 topic, and letting the 16-hour run finish instead of using a subset.
- Long runs were stopped, resumed and moved to the folders described in the README; the repository was reorganised; the generated documents were edited so they say only what was done.

**Where the AI output was wrong, and what happened**
- A wrong Windows command (`source` instead of the PowerShell activation script): reported and corrected.
- An early count of 216 queries for Credit; with the retrained model the correct count is 207 (69 queries x 3 seeds). The count in the files was right.
- The model check in `phase2_attribution.py` raised a false alarm because its tolerance was tighter than the 32-bit values stored in the results file; fixed after a diagnostic run.
- A quick test with `--limit` overwrote a full Phase 2 result; the script was changed so that `--limit` runs write to their own folder, and the full analysis was re-run.
- A long regeneration run ran out of memory; the script was changed to free memory after each query and to resume selected rankings.
- A draft email stated a cause in the authors' code as a fact; it is only our reading of the code, so the wording was changed to say we suspect it.

**What the AI did not do**
- It did not run any experiment, download any data, or produce any result file.
- Conclusions in the results documents were checked against the output files before being kept. Claims that are our interpretation (for example, why GradCFA ranks Income first) are worded as interpretations.

### 4. Session log

| Date | What was asked for | What we did with the answer |
|---|---|---|
| Before 2026-10-06 | Compare candidate papers, check venue and code | Selected GradCFA after checking the venue and the repository |
| 2026-10-06 to 2026-10-10 | Setup errors (`torch` missing, import path, class name, model path) | Installed packages, edited the notebook imports and path, ran the notebook |
| 2026-10-10 | `TypeError` from a missing `return` in `DiPACE.py`; outdated `evaluate()` call | Added the missing line, restarted the kernel, replaced the evaluation cell |
| 2026-10-10 | Seed cell and reproducibility check | Added the seed cell and verified it on two runs |
| 2026-10-11 to 2026-10-12 | Script to run GradCFA over the whole test set, ablation configurations | Ran it on a laptop, compared the summaries with the paper's table |
| 2026-10-18 | Column-order mismatch between notebook and `DataInterface`, retraining script | Ran the check, retrained the model, reran Config A |
| 2026-10-19 to 2026-10-21 | Runs over all 138 test rows, full method with perturbation, ablation grid | Ran them, compared the summaries with the paper's table |
| 2026-10-22 to 2026-10-26 | Phase 2 design and scripts (exact Shapley values, Path-IG, tests T1 to T3) | Ran the diagnostic, the cross-check against `shap`, the analysis on three further seeds, and the regeneration test in several resumed parts |
| 2026-10-27 to 2026-10-29 | 60/20/20 split, validation-only tuning, plausibility sensitivity, plan for the three datasets, hardware question | Ran the scripts, read the outputs, set up the team plan |
| 2026-10-30 to 2026-11-03 | Repository layout, final README, logs, review of the team's Obesity and Fetal Health reports | Moved files, ran a fresh-clone test, corrected the documents |

