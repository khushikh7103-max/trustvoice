"""REPORT TOOL (not part of the web app). Run:  python evaluate.py
Prints and saves the numbers for sections (a) model and (b) LLM of the report.
Set ANTHROPIC_API_KEY first to evaluate the real LLM (about 11 API calls)."""
import json, app, ml, llm

rep = ml.REPORT
print("== (a) PREDICTIVE MODEL ==")
for k in ("baseline_majority", "baseline_keywords", "model", "model_stress"):
    m = rep[k]
    p = "n/a" if m["precision"] is None else f'{m["precision"]:.1%}'
    print(f'{k:20s} precision {p:>7}  recall {m["recall"]:.1%}  false alarms {m["fp"]}  missed {m["fn"]}  cost INR {m["cost"]:,}')
print("threshold", rep["threshold"], "| cost per miss", rep["cost_fn"], "| cost per false alarm", rep["cost_fp"])
print("missed examples:", rep["false_negatives"]); print("false-alarm examples:", rep["false_positives"])
print("\n== (b) LLM ==")
res = llm.run_eval(scrub=app.scrub)
print(json.dumps(res, indent=2))
out = {k: v for k, v in rep.items() if k != "points"}
json.dump({"model": out, "llm": res}, open("report_numbers.json", "w"), indent=2)
print("\nSaved report_numbers.json")
