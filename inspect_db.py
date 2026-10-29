"""Shows where the data is stored and what is inside.   python inspect_db.py"""
import os, sqlite3
path = os.path.abspath(os.environ.get("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.db")))
print("Database file:", path)
if not os.path.exists(path):
    print("Not created yet. Start the app once (python app.py)."); raise SystemExit
c = sqlite3.connect(path); c.row_factory = sqlite3.Row
for t in ("feedback", "pulse", "followups", "summaries"):
    print(f"\n[{t}] {c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")
    for r in c.execute(f"SELECT * FROM {t} LIMIT 3"):
        print("  ", {k: (str(r[k])[:60]) for k in r.keys()})
