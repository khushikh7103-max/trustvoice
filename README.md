# TrustVoice — AI-Powered Anonymous Employee Feedback System (class project)

Flask app: employee form (anonymous) → name/phone/email masking → ML urgent-comment detector → human review queue
(+ 5% blind audit) → LLM theme summaries (validated by code) → human approval → leadership view.
Also: model report with baselines and a cost slider, LLM test harness, 3-year buy/rent/build calculator, risk table.

**All data is synthetic. Do not enter real personal information.**

## 1. Run it in VS Code (Windows / Mac / Linux)
1. Install Python 3.10+ (python.org) and VS Code (with the "Python" extension).
2. VS Code → File → Open Folder → choose `hr-feedback`.
3. Terminal → New Terminal, then:

Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Mac/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass` first.)

4. Set an HR console password before starting the app, then open http://localhost:5000.

Set `HR_PASSWORD` in your terminal (do not put it in source code):

Windows PowerShell:
```powershell
$env:HR_PASSWORD="choose-a-strong-password"
python app.py
```

Mac/Linux:
```bash
export HR_PASSWORD="choose-a-strong-password"
python3 app.py
```

### Enable the real LLM (otherwise a rule-based fallback runs and the LLM test tab says so)
Get an API key at console.anthropic.com, then before `python app.py`:
- Windows PowerShell: `$env:ANTHROPIC_API_KEY="sk-ant-..."`
- Mac/Linux: `export ANTHROPIC_API_KEY="sk-ant-..."`
Optional: `ANTHROPIC_MODEL` (default `claude-haiku-4-5`), `HR_PASSWORD`, `K_MIN` (default 5).

### Where the data is stored
- **Application data:** one SQLite file, `feedback.db`, created automatically in this project folder the first time you run `python app.py`
  (full path is shown in HR console → *Data and storage*). Tables: `feedback`, `pulse`, `followups`, `summaries`.
  Change the location with the `DB_PATH` environment variable.
- **View it:** run `python inspect_db.py`, or install the "SQLite Viewer" extension in VS Code and click `feedback.db`.
  HR can also download everything (masked) via *Export all messages (CSV)*.
- **Sample data:** on first start the app fills the database with ~80 sample messages and weekly pulse answers. Delete `feedback.db` to reset.
- **Model training data:** generated in memory by `ml.py` at start-up; it is not stored in a file.
- **On Render's free plan** the file is wiped on every redeploy/restart. For persistence attach a Render Disk and set `DB_PATH=/var/data/feedback.db`.

### Screens
Submit feedback (concern or idea, live masking preview, receipt code, weekly pulse) · Ideas board · Track a submission (status, HR reply, follow-up) ·
HR console (KPIs, charts, review queue, idea publishing, teams, summaries, data and storage, CSV exports) · Leadership summary (approved summaries + one-click leadership report).

### HR ticket closure and AI-assisted replies

- The HR review queue includes **Close ticket**. Closing a ticket changes its status to `closed`, removes it from the pending review queue, and displays **Closed by HR** when the employee checks their receipt code.
- Each pending ticket includes **Generate AI reply**. The draft is based on the scrubbed feedback and its follow-ups, and is inserted into the reply field for HR to review/edit. Generating a draft does not send it.
- When `ANTHROPIC_API_KEY` is configured, the app uses the configured Anthropic model (`ANTHROPIC_MODEL`, default `claude-haiku-4-5`) to draft a concise, empathetic reply. If the key is missing or the provider is unavailable, a clearly labelled template fallback is used.
- HR must explicitly choose an action (Escalate, No action needed, or Close ticket) to save the reply and update the ticket. The AI draft must be checked for accuracy before sending; it does not claim an investigation or resolution has occurred.

### Predictive urgency triage (HR console)

**Model used:** a scikit-learn `LogisticRegression` binary classifier, fed by a `FeatureUnion` of two TF-IDF text representations:
- **Word TF-IDF (unigrams + bigrams):** captures individual terms and short phrases in context.
- **Character TF-IDF (3–5 character n-grams):** helps with spelling variations, word forms, and slightly altered wording.

The leadership report's **Reviewed by HR** KPI counts tickets whose status is `escalated`, `false_alarm`, or `closed`. Tickets with status `received` or `pending_review` are not counted as reviewed. This keeps the KPI aligned with the actions available in the HR console.

The model outputs a probability that a comment is urgent. The app uses a **0.60 decision threshold** for the urgent flag; the probability is then mapped to Low / Medium / High / Critical tiers. The threshold is deliberately not chosen by minimizing the synthetic cost function alone, because that selected a very low threshold and created many false alarms on the held-out stress set. Training uses `class_weight="balanced"` to account for the synthetic class imbalance. No external pretrained model or LLM is used for this prediction.

**Synthetic benchmark after this update** (not a real-world accuracy claim):
- Held-out synthetic test set at threshold 0.60: **97.5% accuracy**, **95.7% precision**, **84.6% recall**, **0.898 F1** (5 false positives, 20 false negatives).
- Separate template-held-out stress set at threshold 0.60: **97.8% accuracy**, **94.7% precision**, **88.9% recall**, **0.917 F1** (4 false positives, 9 false negatives).
- For comparison, the previous 0.50 threshold scored 97.0% / 97.5% accuracy on those sets, with F1 of 0.881 / 0.906. The 0.60 threshold reduces false alarms in these synthetic benchmarks, but slightly increases missed urgent cases in the main test set.

These numbers are based on generated examples and should be treated only as a demonstration. Real accuracy requires a representative, consented, human-labelled dataset, careful privacy controls, and evaluation for false negatives across different writing styles and groups. Urgency scores should prioritize human review, not make employment decisions or replace HR/safety escalation procedures.

Each message waiting for review gets a **predicted urgency tier** (Critical / High / Medium / Low), a **response target**
(Critical 24 h, High 72 h, Medium 7 days) and a plain-language reason. The review queue is sorted most-urgent first and can be filtered by tier.
- Base tier: the urgent-probability from the ML model (`ml.score`).
- Raised (never lowered) by: an author follow-up or 2+ flagged reports from the same team in the last 3 weeks.
- "Past target" appears when an item has waited longer than its response target (time is known to the week only, by design).
- Dashboard tiles show Critical/High items waiting and a **forecast of urgent messages next week** (weighted average of the last 3 completed weeks).
- *Urgency predictor* card: paste any text to see the urgency tier and key phrases (nothing is saved).
- Employees never see scores or tiers. The tier only decides the order a human reads things; it never triggers action by itself.
- Data is synthetic, so the evaluation figures (`python evaluate.py`) illustrate the method, not real-world performance.

### Quick demo script
1. Submit tab: send a comment, note the receipt code.
2. HR console: sign in using the password configured in the `HR_PASSWORD` environment variable; review the queue, escalate with a reply, and check the receipt on the Submit tab.
3. Try the Ideas board (vote), the weekly pulse, and a follow-up from My receipt.
4. HR console: Generate draft → Approve → see it in Leadership view.

## 2. Put it on a public link (Render, free tier)
1. Create a GitHub account and a new repository; upload this folder (or in VS Code: Source Control → Initialize Repository → Publish to GitHub). `.gitignore` already excludes the DB and secrets.
2. Go to render.com → New → Web Service → connect the repo. (Or New → Blueprint, which reads `render.yaml`.)
3. Settings: Build `pip install -r requirements.txt`, Start `gunicorn app:app --workers 1 --threads 4 --bind 0.0.0.0:$PORT`.
4. Environment variables: `HR_PASSWORD` (choose one), `ANTHROPIC_API_KEY`, `SECRET_KEY` (any long random text).
5. Deploy. You get `https://<name>.onrender.com`. Free instances sleep after ~15 min idle (first load takes ~30–60 s: open it before presenting) and the database resets on redeploy (demo data re-seeds automatically).
Keep your API key only in Render's environment settings; never commit it. Set a spending limit on the key.
Alternatives: Railway, Fly.io, or PythonAnywhere (any host that runs `gunicorn app:app`).

## 3. Numbers for the report (kept out of the app)
Run `python evaluate.py` in the terminal. It prints and saves `report_numbers.json` with model-vs-baseline precision/recall and
error costs, plus LLM fabrication/error rates and failure examples. Risks, human checkpoint, business case and prompt log
belong in the written report only (`prompt_log.md` is the appendix template).

## Files
`app.py` server · `inspect_db.py` shows the stored data · `ml.py` model + synthetic data · `llm.py` LLM + test harness · `evaluate.py` report numbers · `static/index.html` UI · `prompt_log.md` · `render.yaml`/`Procfile` deployment.
