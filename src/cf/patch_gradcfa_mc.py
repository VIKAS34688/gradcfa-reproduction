"""
Patches src/cf/GradCFA_MC.py (run once from the repository root):  python patch_gradcfa_mc.py

Two fixes, both to be listed under "Issues found in the original repo" in the README:
 1. The feature-importance ranking was only a local variable. Now stored in self.importance_ranking.
 2. The function ended with `cf_instances = best_cf_instances`, but best_cf_instances is only updated
    inside the perturbation branch. If the optimisation converged without perturbation, the returned
    counterfactuals could be the unoptimised starting points. Now the final counterfactuals are kept
    unless an earlier perturbation round reached a lower loss.
Safe to run twice (it checks whether the patch is already there).
"""
PATH = "src/cf/GradCFA_MC.py"
raw = open(PATH, "r", encoding="utf-8", newline="").read()
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")

if "self.importance_ranking" in s:
    print("Already patched.")
    raise SystemExit

old1 = "        ranked_importance = [importance for _, importance in importance_ranking]\n"
new1 = old1 + "        self.importance_ranking = importance_ranking\n"
old2 = "        cf_instances = best_cf_instances\n"
new2 = (
    "        cf_instances = cf_instances.detach()\n"
    "        with torch.no_grad():\n"
    "            _cur_loss = self.compute_loss(query_instance, cf_instances, k).item()\n"
    "        if best_loss != float('inf') and _cur_loss > float(best_loss):\n"
    "            cf_instances = best_cf_instances\n"
)
assert s.count(old1) == 1, "ranked_importance line not found exactly once"
assert s.count(old2) == 1, "best_cf_instances line not found exactly once"
s = s.replace(old1, new1).replace(old2, new2)
if crlf:
    s = s.replace("\n", "\r\n")
open(PATH, "w", encoding="utf-8", newline="").write(s)
print("Patched", PATH)
