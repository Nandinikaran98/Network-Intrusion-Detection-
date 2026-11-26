#!/usr/bin/env python3
"""
app.py - web form for the Network Intrusion Detection System (Flask).

    python app.py        ->  open http://127.0.0.1:5000 in your browser

Also offers a JSON API:  POST /api/predict   {"proto": "tcp", "sbytes": 500, ...}
"""
import os

from flask import Flask, jsonify, render_template_string, request

import common as C

MODEL_DIR = os.environ.get("NIDS_MODEL_DIR", os.path.join("outputs", "models"))
app = Flask(__name__)
MODEL, META = C.load_artifacts(MODEL_DIR)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network Intrusion Detection System</title>
<style>
  :root { --bg:#f4f6fb; --card:#fff; --ink:#1d2433; --muted:#667085; --line:#dfe3ec;
          --accent:#1d3557; --good:#1b7f5c; --bad:#c62839; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  header { background:var(--accent); color:#fff; padding:22px 16px; text-align:center; }
  header h1 { margin:0 0 4px; font-size:1.55rem; } header p { margin:0; opacity:.8; font-size:.92rem; }
  main { max-width:900px; margin:22px auto; padding:0 16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin-bottom:18px; }
  h2 { margin:0 0 14px; font-size:1.05rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; }
  label { display:block; font-size:.82rem; font-weight:600; margin-bottom:4px; }
  label small { font-weight:400; color:var(--muted); }
  input, select { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:8px; font-size:.95rem; background:#fff; }
  input:focus, select:focus { outline:2px solid #9db4d6; }
  .row { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
  button { padding:10px 18px; border-radius:8px; border:1px solid var(--accent); background:var(--accent); color:#fff; font-size:.95rem; cursor:pointer; }
  button.ghost { background:#fff; color:var(--accent); }
  .result { text-align:center; padding:26px 16px; border-radius:12px; border:2px solid; }
  .result.good { border-color:var(--good); background:#ecf8f3; color:var(--good); }
  .result.bad { border-color:var(--bad); background:#fdeef0; color:var(--bad); }
  .result .big { font-size:1.9rem; font-weight:800; margin:4px 0; }
  .error { background:#fff3cd; border:1px solid #f0d98a; padding:12px 14px; border-radius:8px; margin-bottom:16px; }
  .note { color:var(--muted); font-size:.82rem; }
  .bar { height:10px; background:#e7eaf2; border-radius:6px; overflow:hidden; margin:10px auto 0; max-width:360px; }
  .bar span { display:block; height:100%; background:currentColor; }
</style></head><body>
<header><h1>Network Intrusion Detection System</h1>
<p>Model: {{ meta.model_name }} &middot; trained on UNSW-NB15 network traffic</p></header>
<main>
{% if error %}<div class="error"><b>Input problem:</b> {{ error }}</div>{% endif %}
{% if result %}
  <div class="card result {{ 'bad' if result.label else 'good' }}">
    <div>Predicted class</div>
    <div class="big">{{ result.prediction }}</div>
    <div>Attack probability: <b>{{ '%.1f' % (result.attack_probability * 100) }}%</b></div>
    <div class="bar"><span style="width:{{ '%.1f' % (result.attack_probability * 100) }}%"></span></div>
  </div>
{% endif %}
<form method="post" class="card" id="f">
  <h2>Network traffic features</h2>
  <div class="grid">
  {% for key in meta.key_features %}
    {% set info = meta.feature_info[key] %}
    <div>
      <label for="{{ key }}">{{ info[0] }} <small>({{ key }})</small></label>
      {% if key in meta.categorical %}
        <select name="{{ key }}" id="{{ key }}">
          {% for c in meta.choices[key] %}
            <option value="{{ c }}" {{ 'selected' if (values.get(key, meta.defaults[key]) == c) }}>{{ c }}</option>
          {% endfor %}
        </select>
      {% else %}
        <input type="number" step="any" name="{{ key }}" id="{{ key }}"
               value="{{ values.get(key, '') }}" placeholder="default {{ '%g' % meta.defaults[key] }}">
      {% endif %}
    </div>
  {% endfor %}
  </div>
  <p class="note">Empty fields use the typical (median) value of the training data. The other
  {{ meta.features|length - meta.key_features|length }} model features are also set to their typical values.</p>
  <div class="row">
    <button type="submit">Classify traffic</button>
    <button type="button" class="ghost" onclick="fillExample('normal')">Load normal example</button>
    <button type="button" class="ghost" onclick="fillExample('attack')">Load attack example</button>
    <button type="button" class="ghost" onclick="clearForm()">Clear</button>
  </div>
</form>
<p class="note">Educational project. The prediction depends on the trained model and the supplied
input; it is not a replacement for a production intrusion detection system.</p>
</main>
<script>
  const EXAMPLES = {{ meta.examples | tojson }};
  function fillExample(kind) {
    const ex = EXAMPLES[kind]; if (!ex) return;
    for (const [k, v] of Object.entries(ex)) { const el = document.getElementById(k); if (el) el.value = v; }
  }
  function clearForm() {
    document.querySelectorAll('#f input').forEach(i => i.value = '');
    document.querySelectorAll('#f select').forEach(s => s.selectedIndex = 0);
  }
</script>
</body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
    if request.method == "POST":
        values = {k: v.strip() for k, v in request.form.items() if v.strip()}
        try:
            result = C.predict_records([values], MODEL, META)[0]
        except ValueError as exc:
            error = str(exc)
    return render_template_string(PAGE, meta=META, result=result, error=error, values=values)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with feature names as keys."), 400
    try:
        return jsonify(C.predict_records([payload], MODEL, META)[0])
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
