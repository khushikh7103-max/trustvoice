import os, re, json, sqlite3, secrets, hashlib, time, random, datetime
from flask import Flask, request, jsonify, session, send_from_directory, Response
import ml, llm

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("DB_PATH", os.path.join(BASE, "feedback.db"))
K_MIN = int(os.environ.get("K_MIN", "5"))            # k-anonymity floor for any group statistic
HR_PASSWORD = os.environ.get("HR_PASSWORD", "hr-demo").strip()
AUDIT_RATE = float(os.environ.get("AUDIT_RATE", "0.05"))  # share of UNflagged comments sent to a blind human audit
DEPARTMENTS = ["Engineering", "Finance", "Operations", "Sales", "HR & Admin"]
EVAL_FILE = os.path.join(BASE, "eval_results.json")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-change-me")
ml.train()

# ---------------------------------------------------------------- db
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, token_hash TEXT, department TEXT,
          week TEXT, text TEXT, score REAL, flagged INTEGER, audit INTEGER DEFAULT 0, status TEXT, hr_note TEXT, hr_reply TEXT,
          kind TEXT DEFAULT 'concern', published INTEGER DEFAULT 0, votes INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS pulse(id INTEGER PRIMARY KEY AUTOINCREMENT, department TEXT, week TEXT, mood INTEGER);
        CREATE TABLE IF NOT EXISTS followups(id INTEGER PRIMARY KEY AUTOINCREMENT, feedback_id INTEGER, week TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS summaries(id INTEGER PRIMARY KEY AUTOINCREMENT, department TEXT, created TEXT,
          n INTEGER, mode TEXT, body TEXT, status TEXT);""")
        for col in ("kind TEXT DEFAULT 'concern'", "published INTEGER DEFAULT 0", "votes INTEGER DEFAULT 0"):   # upgrade older databases
            try:
                c.execute("ALTER TABLE feedback ADD COLUMN " + col)
            except sqlite3.OperationalError:
                pass

# ---------------------------------------------------------------- privacy: scrubbing
SAFE_CAPS = {"I", "HR", "IT", "POSH", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
             "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "India", "Please", "Thanks",
             "Honestly", "FYI", "Just", "Not", "Sharing", "Hope", "That", "My", "Our", "The", "We", "It", "This", "He", "She", "They", "A", "In", "On"}
def scrub(text):
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    t = re.sub(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)", "[PHONE]", t)
    t = re.sub(r"https?://\S+", "[LINK]", t)
    t = re.sub(r"\b[A-Z]{2,4}-?\d{3,7}\b", "[EMP-ID]", t)
    t = re.sub(r"\b(Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", "[NAME]", t)
    # Mask likely personal names before saving. Context cues catch names after role
    # words; a common-name list catches lowercase names such as "rahul from finance".
    name_cues = r"(?i)\b(?:named|called|from|by|with|against|reported|complained about|manager|supervisor|colleague|director|lead|hr contact)\s+([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,2})"
    t = re.sub(name_cues, lambda m: m.group(0)[:len(m.group(0))-len(m.group(1))] + "[NAME]", t)
    common_names = (
        "aarav,aadhya,aarohi,abhay,aditi,aditya,akash,akshay,aman,amita,amit,ananya,anil,anjali,ankit,arjun,"
        "arnav,arun,ashish,ayesha,deepak,dev,diya,farhan,gaurav,ishita,jai,jaanvi,kabir,kajal,karan,kavya,"
        "khushi,kunal,lakshmi,manish,meera,mohit,neha,nikhil,nisha,pooja,prachi,pradeep,priya,rahul,raj,"
        "rajesh,rakesh,rhea,riya,rohan,rohit,roshni,sahil,sakshi,sameer,sanjay,sara,shreya,simran,sneha,"
        "sohan,sonal,suhani,sumit,suraj,tanvi,tarun,varun,vikas,vijay,yash,zoya,john,jane,michael,david,"
        "emma,olivia,james,robert,maria,daniel,alex,sam,sophia"
    ).split(",")
    for name in sorted(set(common_names), key=len, reverse=True):
        t = re.sub(r"(?i)\b" + re.escape(name) + r"\b", "[NAME]", t)
    words, out = t.split(" "), []
    for i, w in enumerate(words):
        core = w.strip(".,;:!?()\\\"'")
        prev = words[i - 1] if i else ""
        if i and not prev.endswith((".", "!", "?", ":")) and re.fullmatch(r"[A-Z][a-z]{2,}", core) and core not in SAFE_CAPS:
            w = w.replace(core, "[NAME]")
        out.append(w)
    return " ".join(out)

# ---------------------------------------------------------------- core pipeline
def week_str(d=None):
    d = d or datetime.date.today()
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"          # timestamps are coarsened to the week to reduce re-identification by timing

def weeks_back(n=8):
    today = datetime.date.today()
    return [week_str(today - datetime.timedelta(days=7 * i)) for i in range(n - 1, -1, -1)]

def tags(text, k=2):
    t = text.lower()
    hits = sorted(((sum(w in t for w in ws), th) for th, ws in llm.KEYWORDS.items()), reverse=True)
    return [th for n, th in hits[:k] if n > 0]

def process(text, dept, week=None, rng=None, kind="concern", published=0, votes=0):
    rng = rng or random
    clean = scrub(text.strip())
    p = ml.score(clean)
    flagged = int(p >= ml.THRESHOLD)
    audit = int((not flagged) and rng.random() < AUDIT_RATE)
    status = "pending_review" if (flagged or audit) else "received"
    token = "FB-" + secrets.token_hex(4).upper()
    with db() as c:
        c.execute("INSERT INTO feedback(token_hash,department,week,text,score,flagged,audit,status,kind,published,votes) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (hashlib.sha256(token.encode()).hexdigest(), dept, week or week_str(), clean, p, flagged, audit, status, kind, published, votes))
    return token, clean, p, flagged

def seed():
    with db() as c:
        if c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]:
            return
    rng = random.Random(7)
    for item in ml.sample_comments(80, seed=7, split="test"):
        d = rng.choices(DEPARTMENTS, weights=[5, 3, 3, 3, 1])[0]
        wk = week_str(datetime.date.today() - datetime.timedelta(days=rng.randint(0, 55)))
        idea = item["theme"] in ("Facilities & tools", "Career growth", "Culture & recognition") and rng.random() < .6
        process(item["text"], d, wk, rng, kind="idea" if idea else "concern",
                published=int(idea and rng.random() < .7), votes=rng.randint(0, 14) if idea else 0)
    with db() as c:
        for i, wk in enumerate(weeks_back(8)):
            for dep in DEPARTMENTS:
                for _ in range(rng.randint(2, 4)):
                    mood = min(5, max(1, round(rng.gauss(3.9 - (0.5 if i in (4, 5) else 0) - (0.3 if dep == "Operations" else 0), 1.0))))
                    c.execute("INSERT INTO pulse(department,week,mood) VALUES(?,?,?)", (dep, wk, mood))

init_db()
if os.environ.get("SEED_DEMO", "1") == "1":
    seed()

# simple in-memory rate limit (nothing about the sender is written to disk)
_hits = {}
def rate_ok():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    k = hashlib.sha256((ip + app.secret_key).encode()).hexdigest()
    now = time.time()
    _hits[k] = [t for t in _hits.get(k, []) if now - t < 3600]
    if len(_hits[k]) >= 15:
        return False
    _hits[k].append(now)
    return True

def hr_only(f):
    def w(*a, **kw):
        if not session.get("hr"):
            return jsonify(error="HR login required"), 401
        return f(*a, **kw)
    w.__name__ = f.__name__
    return w

# ---------------------------------------------------------------- predictive triage
def _age_weeks(week):
    y, w = week.split("-W")
    mon = datetime.date.fromisocalendar(int(y), int(w), 1)
    return max(0, (datetime.date.today() - mon).days // 7)

def triage_pending(c):
    """Every message waiting for human review, with a predicted urgency tier, most urgent first."""
    rows = c.execute("SELECT id,department,week,text,kind,score FROM feedback WHERE status='pending_review'").fetchall()
    recent = weeks_back(3)
    flagged = c.execute(f"SELECT id,department,text FROM feedback WHERE flagged=1 AND week IN ({','.join('?' * len(recent))})", recent).fetchall()
    ftheme = [(r["id"], r["department"]) for r in flagged]
    items = []
    for r in rows:
        fu = [x["text"] for x in c.execute("SELECT text FROM followups WHERE feedback_id=? ORDER BY id", (r["id"],)).fetchall()]
        similar = sum(1 for i, d in ftheme if i != r["id"] and d == r["department"])
        u = ml.urgency(r["text"], p=r["score"], followups=len(fu), similar=similar)
        age = _age_weeks(r["week"])
        u["overdue"] = bool(u["sla_hours"] and age >= 1 and age * 168 > u["sla_hours"])
        items.append(dict(r) | {"tags": tags(r["text"]), "followups": fu, "urgency": u})
    items.sort(key=lambda i: (-i["urgency"]["rank"], not i["urgency"]["overdue"], -i["score"]))
    return items

def forecast_urgent(c):
    """Expected number of urgent (model-flagged) messages next week: weighted average of the last 3 completed weeks."""
    wk = weeks_back(8)[-4:-1]
    n = [c.execute("SELECT COUNT(*) FROM feedback WHERE flagged=1 AND week=?", (w,)).fetchone()[0] for w in wk]
    return round(sum(a * b for a, b in zip(n, (1, 2, 3))) / 6, 1), n

# ---------------------------------------------------------------- routes: employee
@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/config")
def config():
    return jsonify(departments=DEPARTMENTS, k_min=K_MIN, llm_mode="llm" if os.environ.get("ANTHROPIC_API_KEY") else "fallback", llm_model=llm.MODEL)

@app.post("/api/submit")
def submit():
    if not rate_ok():
        return jsonify(error="Too many submissions from this network. Try later."), 429
    d = request.get_json(force=True)
    text, dept = (d.get("text") or "").strip(), d.get("department")
    kind = d.get("kind") if d.get("kind") in ("concern", "idea") else "concern"
    if len(text) < 15 or len(text) > 1500 or dept not in DEPARTMENTS:
        return jsonify(error="Write 15-1500 characters and choose a department."), 400
    token, clean, p, flagged = process(text, dept, kind=kind)
    # the response never reveals the model score to the employee
    return jsonify(receipt=token, stored_text=clean, note="Keep this receipt code. It is the only way to check for a reply. We do not store your IP or name.")

@app.get("/api/status/<token>")
def status(token):
    h = hashlib.sha256(token.strip().upper().encode()).hexdigest()
    with db() as c:
        r = c.execute("SELECT status,hr_reply,week FROM feedback WHERE token_hash=?", (h,)).fetchone()
    if not r:
        return jsonify(error="Receipt not found"), 404
    label = {"received": "Received", "pending_review": "Under HR review", "escalated": "Escalated to the HR case team",
             "false_alarm": "Reviewed - no urgent action needed", "closed": "Closed by HR"}.get(r["status"], "Under review")
    return jsonify(status=label, reply=r["hr_reply"], week=r["week"])

@app.post("/api/preview")
def preview():   # shows the author exactly what will be stored; nothing is saved or scored
    return jsonify(masked=scrub((request.get_json(force=True).get("text") or "")[:1500]))

@app.post("/api/pulse")
def pulse():
    d = request.get_json(force=True)
    if d.get("department") not in DEPARTMENTS or d.get("mood") not in (1, 2, 3, 4, 5):
        return jsonify(error="Choose a team and a mood."), 400
    if not rate_ok():
        return jsonify(error="Too many requests from this network."), 429
    with db() as c:
        c.execute("INSERT INTO pulse(department,week,mood) VALUES(?,?,?)", (d["department"], week_str(), d["mood"]))
    return jsonify(ok=True)

@app.post("/api/followup")
def followup():
    d = request.get_json(force=True)
    text = (d.get("text") or "").strip()
    if len(text) < 3 or len(text) > 800 or not rate_ok():
        return jsonify(error="Write 3-800 characters."), 400
    h = hashlib.sha256((d.get("token") or "").strip().upper().encode()).hexdigest()
    with db() as c:
        r = c.execute("SELECT id FROM feedback WHERE token_hash=?", (h,)).fetchone()
        if not r:
            return jsonify(error="Receipt not found"), 404
        c.execute("INSERT INTO followups(feedback_id,week,text) VALUES(?,?,?)", (r["id"], week_str(), scrub(text)))
        c.execute("UPDATE feedback SET status='pending_review' WHERE id=?", (r["id"],))   # a human must see the follow-up
    return jsonify(ok=True)

@app.get("/api/ideas")
def ideas():
    with db() as c:
        rows = c.execute("SELECT id,text,votes FROM feedback WHERE kind='idea' AND published=1 ORDER BY votes DESC, id DESC").fetchall()
    return jsonify(items=[dict(r) for r in rows])

@app.post("/api/ideas/<int:iid>/vote")
def vote(iid):
    if not rate_ok():
        return jsonify(error="Too many requests"), 429
    with db() as c:
        c.execute("UPDATE feedback SET votes=votes+1 WHERE id=? AND kind='idea' AND published=1", (iid,))
        v = c.execute("SELECT votes FROM feedback WHERE id=?", (iid,)).fetchone()
    return jsonify(votes=v["votes"] if v else 0)

# ---------------------------------------------------------------- routes: HR
@app.post("/api/hr/login")
def login():
    if not HR_PASSWORD:
        return jsonify(error="HR sign-in is not configured on this server. Set the HR_PASSWORD environment variable."), 503
    if secrets.compare_digest((request.get_json(force=True).get("password") or ""), HR_PASSWORD):
        session["hr"] = True
        return jsonify(ok=True)
    return jsonify(error="Wrong password"), 403

@app.post("/api/hr/logout")
def logout():
    session.clear()
    return jsonify(ok=True)

@app.get("/api/hr/queue")
@hr_only
def queue():
    with db() as c:
        items = triage_pending(c)
        ideas_pending = c.execute("SELECT id,department,text FROM feedback WHERE kind='idea' AND published=0 AND status IN ('received','false_alarm') ORDER BY id DESC LIMIT 20").fetchall()
    return jsonify(items=items, ideas=[dict(r) | {"tags": tags(r["text"])} for r in ideas_pending])

@app.post("/api/hr/predict")
@hr_only
def predict_demo():   # try the urgency model on any text; nothing is stored
    text = scrub((request.get_json(force=True).get("text") or "").strip()[:1500])
    if len(text) < 10:
        return jsonify(error="Write at least 10 characters."), 400
    return jsonify(masked=text, urgency=ml.urgency(text))

@app.post("/api/hr/idea")
@hr_only
def idea_decision():
    d = request.get_json(force=True)
    with db() as c:
        if d.get("publish"):
            c.execute("UPDATE feedback SET published=1 WHERE id=? AND kind='idea' AND status IN ('received','false_alarm')", (d.get("id"),))
        else:
            c.execute("UPDATE feedback SET published=-1 WHERE id=? AND kind='idea'", (d.get("id"),))
    return jsonify(ok=True)

@app.get("/api/hr/me")
def hr_me():
    return jsonify(hr=bool(session.get("hr")))

@app.get("/api/hr/storage")
@hr_only
def storage():
    with db() as c:
        counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("feedback", "pulse", "followups", "summaries")}
    path = os.path.abspath(DB)
    return jsonify(path=path, exists=os.path.exists(path), size_kb=round(os.path.getsize(path) / 1024, 1) if os.path.exists(path) else 0,
                   counts=counts, training_data="Model training data is generated in memory by ml.py at start-up and is not stored in any file.")

@app.get("/api/hr/export-all.csv")
@hr_only
def export_all():
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "type", "team", "week", "status", "masked_text", "hr_reply"])
    with db() as c:
        for r in c.execute("SELECT id,kind,department,week,status,text,hr_reply FROM feedback ORDER BY id").fetchall():
            w.writerow([r["id"], r["kind"], r["department"], r["week"], r["status"], r["text"], r["hr_reply"] or ""])
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=all_messages_masked.csv"})

@app.get("/api/hr/export.csv")
@hr_only
def export_csv():
    with db() as c:
        rows = []
        for dep in DEPARTMENTS:
            n = c.execute("SELECT COUNT(*) FROM feedback WHERE department=?", (dep,)).fetchone()[0]
            if n < K_MIN:
                rows.append(f"{dep},suppressed,,,")
                continue
            es = c.execute("SELECT COUNT(*) FROM feedback WHERE department=? AND status='escalated'", (dep,)).fetchone()[0]
            m = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse WHERE department=?", (dep,)).fetchone()
            mood = f"{m[0]:.2f}" if m[1] >= K_MIN else "suppressed"
            rows.append(f"{dep},{n},{es},{mood},{m[1]}")
    body = "team,comments,escalated,avg_mood,pulse_responses\n" + "\n".join(rows) + "\n"
    return Response(body, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=team_summary.csv"})

@app.post("/api/hr/draft-reply")
@hr_only
def draft_reply():
    """Generate a reply draft for HR to review; never sends or saves the draft automatically."""
    d = request.get_json(force=True)
    try:
        fid = int(d.get("id"))
    except (TypeError, ValueError):
        return jsonify(error="Invalid ticket id"), 400
    with db() as c:
        row = c.execute("SELECT id,department,text,kind,status FROM feedback WHERE id=?", (fid,)).fetchone()
        followups = [x["text"] for x in c.execute("SELECT text FROM followups WHERE feedback_id=? ORDER BY id", (fid,)).fetchall()]
    if not row:
        return jsonify(error="Ticket not found"), 404
    if row["status"] != "pending_review":
        return jsonify(error="Only tickets awaiting HR review can have a reply drafted"), 409
    result = llm.generate_hr_reply(row["text"], followups, row["department"], row["kind"])
    return jsonify(draft=result["draft"], mode=result["mode"], note=result.get("note"))

@app.post("/api/hr/review")
@hr_only
def review():
    d = request.get_json(force=True)
    if d.get("decision") not in ("escalate", "false_alarm", "close"):
        return jsonify(error="bad decision"), 400
    status = {"escalate": "escalated", "false_alarm": "false_alarm", "close": "closed"}[d["decision"]]
    with db() as c:
        cur = c.execute("UPDATE feedback SET status=?,hr_note=?,hr_reply=? WHERE id=? AND status='pending_review'",
                  (status, (d.get("note") or "")[:500], (d.get("reply") or "")[:1000] or None, d.get("id")))
        if cur.rowcount != 1:
            return jsonify(error="Ticket was already handled or could not be found"), 409
        if d.get("publish") and status == "false_alarm":     # a human reviewed it and judged it safe to share
            c.execute("UPDATE feedback SET published=1 WHERE id=? AND kind='idea'", (d.get("id"),))
    return jsonify(ok=True, status=status)

@app.get("/api/hr/dashboard")
@hr_only
def dashboard():
    with db() as c:
        total = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        depts = []
        for dep in DEPARTMENTS:
            n = c.execute("SELECT COUNT(*) FROM feedback WHERE department=?", (dep,)).fetchone()[0]
            if n < K_MIN:
                depts.append({"department": dep, "suppressed": True})
                continue
            fl = c.execute("SELECT COUNT(*) FROM feedback WHERE department=? AND flagged=1", (dep,)).fetchone()[0]
            es = c.execute("SELECT COUNT(*) FROM feedback WHERE department=? AND status='escalated'", (dep,)).fetchone()[0]
            depts.append({"department": dep, "n": n, "flagged": fl, "escalated": es})
        pend = c.execute("SELECT COUNT(*) FROM feedback WHERE status='pending_review'").fetchone()[0]
        esc_n = c.execute("SELECT COUNT(*) FROM feedback WHERE status='escalated'").fetchone()[0]
        wk = weeks_back(8)
        volume = [{"week": w, "n": c.execute("SELECT COUNT(*) FROM feedback WHERE week=?", (w,)).fetchone()[0]} for w in wk]
        trend = []
        for w in wk:
            m = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse WHERE week=?", (w,)).fetchone()
            trend.append({"week": w, "avg": round(m[0], 2)} if m[1] >= K_MIN else {"week": w, "suppressed": True})
        mood_depts = []
        for dep in DEPARTMENTS:
            m = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse WHERE department=?", (dep,)).fetchone()
            mood_depts.append({"department": dep, "avg": round(m[0], 2)} if m[1] >= K_MIN else {"department": dep, "suppressed": True})
        allm = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse").fetchone()
        drafts = c.execute("SELECT * FROM summaries WHERE status='draft' ORDER BY id DESC").fetchall()
        counts = {l: 0 for l in ml.LEVELS}
        for it in triage_pending(c):
            counts[it["urgency"]["level"]] += 1
        fc, fc_hist = forecast_urgent(c)
    return jsonify(urgency_counts=counts, forecast=fc, forecast_history=fc_hist, total=total, pending=pend, escalated=esc_n, volume=volume, trend=trend, mood_depts=mood_depts,
                   mood_avg=round(allm[0], 2) if allm[1] >= K_MIN else None, departments=depts, drafts=[dict(r) | {"body": json.loads(r["body"])} for r in drafts])

@app.post("/api/hr/summarize")
@hr_only
def summarize():
    dep = (request.get_json(force=True).get("department") or "All")
    with db() as c:
        if dep == "All":
            rows = c.execute("SELECT text FROM feedback ORDER BY id DESC LIMIT 60").fetchall()
        else:
            rows = c.execute("SELECT text FROM feedback WHERE department=? ORDER BY id DESC LIMIT 60", (dep,)).fetchall()
    texts = [r["text"] for r in rows]
    if len(texts) < K_MIN:
        return jsonify(error=f"Only {len(texts)} comments - need at least {K_MIN} to protect anonymity."), 400
    out = llm.summarize(texts)
    body = {"themes": out["themes"], "dropped_quotes": out["dropped_quotes"], "bad_themes": out["bad_themes"], "error": out["error"]}
    with db() as c:
        c.execute("INSERT INTO summaries(department,created,n,mode,body,status) VALUES(?,?,?,?,?,'draft')",
                  (dep, datetime.date.today().isoformat(), len(texts), out["mode"], json.dumps(body)))
    return jsonify(ok=True)

@app.post("/api/hr/summary")
@hr_only
def summary_decision():
    d = request.get_json(force=True)
    with db() as c:
        if d.get("decision") == "approve":
            c.execute("UPDATE summaries SET status='approved' WHERE id=?", (d.get("id"),))
        else:
            c.execute("DELETE FROM summaries WHERE id=? AND status='draft'", (d.get("id"),))
    return jsonify(ok=True)

@app.get("/api/leadership")
def leadership():   # only human-approved summaries are visible here
    with db() as c:
        rows = c.execute("SELECT department,created,n,mode,body FROM summaries WHERE status='approved' ORDER BY id DESC").fetchall()
    return jsonify(items=[dict(r) | {"body": json.loads(r["body"])} for r in rows])

@app.get("/api/leadership/report")
def leadership_report():
    """Aggregate-only report for leadership: k-anonymity applied to every group figure,
    approved summaries only, no individual messages and no follow-ups."""
    with db() as c:
        total = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        kinds = {k: c.execute("SELECT COUNT(*) FROM feedback WHERE kind=?", (k,)).fetchone()[0] for k in ("concern", "idea")}
        # Count every ticket that has received a human decision, including closed tickets.
        # 'received' and 'pending_review' are not yet reviewed by HR.
        handled = c.execute("SELECT COUNT(*) FROM feedback WHERE status IN ('escalated','false_alarm','closed')").fetchone()[0]
        wk = weeks_back(8)
        volume = [{"week": w, "n": c.execute("SELECT COUNT(*) FROM feedback WHERE week=?", (w,)).fetchone()[0]} for w in wk]
        trend = []
        for w in wk:
            m = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse WHERE week=?", (w,)).fetchone()
            trend.append({"week": w, "avg": round(m[0], 2)} if m[1] >= K_MIN else {"week": w, "suppressed": True})
        allm = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse").fetchone()
        teams = []
        for dep in DEPARTMENTS:
            n = c.execute("SELECT COUNT(*) FROM feedback WHERE department=?", (dep,)).fetchone()[0]
            m = c.execute("SELECT AVG(mood), COUNT(*) FROM pulse WHERE department=?", (dep,)).fetchone()
            if n < K_MIN:
                teams.append({"department": dep, "suppressed": True})
            else:
                teams.append({"department": dep, "n": n, "avg": round(m[0], 2) if m[1] >= K_MIN else None})
        rows = c.execute("SELECT department,created,n,mode,body FROM summaries WHERE status='approved' ORDER BY id DESC").fetchall()
    # top themes across approved summaries (counts only; no quotes in the roll-up)
    tally = {}
    for r in rows:
        for t in json.loads(r["body"]).get("themes", []):
            tally[t["theme"]] = tally.get(t["theme"], 0) + int(t.get("count", 0))
    top = [{"theme": k, "count": v} for k, v in sorted(tally.items(), key=lambda kv: -kv[1])[:5]]
    return jsonify(generated=datetime.date.today().isoformat(), k_min=K_MIN, total=total, concerns=kinds["concern"], ideas=kinds["idea"],
                   handled=handled, mood_avg=round(allm[0], 2) if allm[1] >= K_MIN else None, volume=volume, trend=trend, teams=teams,
                   top_themes=top, summaries=[dict(r) | {"body": json.loads(r["body"])} for r in rows])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
