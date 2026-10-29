# Appendix: Prompt Log

## A. Prompts shipped inside the app

### A1. Theme-summary system prompt (llm.py → SYSTEM_PROMPT)
Purpose: turn scrubbed comments into draft themes for HR.
Design choices: comments wrapped in `<comment>` tags and declared as data (prompt-injection defence); fixed theme list;
verbatim quotes only (so code can verify them); "return {"themes":[]}" escape hatch for noise; JSON only.
Validation in code: unknown theme → dropped; quote not found in any source comment → removed and counted.
Full text: see the "LLM test" tab or `llm.py`.

## B. Prompts used while building (fill in with YOUR real prompts and outcomes)
| # | Tool / model | Prompt (verbatim) | What it produced | What I changed or rejected, and why |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

Suggested entries to record honestly: prompt that generated the project code; prompt used to design the synthetic data;
prompts used to write the risk table; any prompt where the AI was wrong and you corrected it (e.g. over-optimistic metrics,
a scrubber that missed names).

## C. Iterations of the in-app prompt
| Version | Change | Reason | Effect on eval |
|---|---|---|---|
| v1 | "Summarise these comments" | baseline | (run eval, record) |
| v2 | Added fixed theme list + JSON schema | free-text themes were inconsistent | |
| v3 | Added "comments are data", verbatim-quote rule | injection + fabricated quotes | |
