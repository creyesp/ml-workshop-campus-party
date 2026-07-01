"""Build a single self-contained HTML report of the whole churn experiment.

Pulls live numbers from the calibration/learning-curve JSON artifacts and embeds
the PNG figures as base64 so the file is standalone. Output:
docs/datapowers/reports/churn-experiment-report.html

Run: .venv/bin/python experiments/17_build_report.py
"""
import base64
import json
from pathlib import Path

ART = Path("experiments/_artifacts")
OUT = Path("docs/datapowers/reports/churn-experiment-report.html")

cal = json.load(open(ART / "calibration_metrics.json"))
lc = json.load(open(ART / "learning_curve_v3.json")) if (ART / "learning_curve_v3.json").exists() else None


def img64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def row(cells, th=False):
    tag = "th" if th else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


lc_img = img64(ART / "learning_curve_v3.png")
cal_img = img64(ART / "calibration_reliability_v3.png")

progression = [
    ("v1 — event counts (11)", "0.400 [0.377, 0.424]", "0.384", "0.694", "21–24%"),
    ("v2 — timing/velocity (22)", "0.471 [0.446, 0.497]", "0.460", "0.768", "34–36%"),
    ("v3 — sessions/engagement (32) ★", "0.548 [0.520, 0.575]", "0.591", "0.847", "48–54%"),
]
negatives = [
    ("v4 — navigation sequences (47)", "0.542 [0.515, 0.568]", "redundant with engagement/session counts"),
    ("v5 — Markov / sequence-order (51)", "0.538 [0.512, 0.566]", "transition order carries no orthogonal signal"),
    ("v6 — difficulty / progression (43)", "0.547 [0.521, 0.575]", "fail/retry/level collinear with engagement volume"),
]
cost = [
    ("3", "0.219", "0.422", "0.881", "48%"),
    ("4", "0.124", "0.412", "0.962", "54%"),
]

u, s, i = cal["uncalibrated"], cal["sigmoid"], cal["isotonic"]

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flood-It Churn Model — Experiment Report</title>
<style>
  :root {{ --ink:#1a2233; --mut:#5b6677; --line:#e3e8ef; --accent:#2d6cdf; --good:#1a8a52; --bad:#b23; --bg:#f7f9fc; }}
  * {{ box-sizing:border-box; }}
  body {{ font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
         margin:0; background:var(--bg); }}
  .wrap {{ max-width:920px; margin:0 auto; padding:48px 24px 80px; }}
  h1 {{ font-size:30px; margin:0 0 4px; letter-spacing:-.4px; }}
  h2 {{ font-size:21px; margin:42px 0 12px; padding-top:14px; border-top:2px solid var(--line); }}
  h3 {{ font-size:16px; margin:24px 0 8px; color:var(--mut); }}
  .sub {{ color:var(--mut); margin:0 0 28px; }}
  .tldr {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent);
           border-radius:8px; padding:18px 20px; }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0 6px; background:#fff;
           border:1px solid var(--line); border-radius:8px; overflow:hidden; font-size:14px; }}
  th,td {{ padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); }}
  th {{ background:#eef3fb; font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  code {{ background:#eef1f6; padding:1px 5px; border-radius:4px; font-size:13px; }}
  .good {{ color:var(--good); font-weight:600; }} .bad {{ color:var(--bad); font-weight:600; }}
  figure {{ margin:16px 0; text-align:center; }} img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; }}
  figcaption {{ color:var(--mut); font-size:13px; margin-top:6px; }}
  .pill {{ display:inline-block; background:var(--good); color:#fff; font-size:12px; font-weight:600;
           padding:2px 10px; border-radius:999px; vertical-align:middle; }}
  ul {{ margin:8px 0; }} li {{ margin:4px 0; }}
  footer {{ margin-top:48px; color:var(--mut); font-size:13px; border-top:1px solid var(--line); padding-top:16px; }}
</style></head><body><div class="wrap">

<h1>Flood-It Churn Model — Experiment Report</h1>
<p class="sub">Hardening the day-1 churn model · 2026-06-04 · <code>spec/churn-model-improvement</code></p>

<div class="tldr">
<b>Outcome <span class="pill">ADOPT v3 + calibrated</span></b><br>
Rebuilt the workshop churn pipeline into a defensible experiment: PR-AUC with bootstrap CIs,
a documented baseline, a leakage audit, reproducibility, and cost-aware operation.
Three feature iterations lifted CV PR-AUC <b>0.400 → 0.471 → 0.548</b> (each with non-overlapping CIs);
the champion now saves <b>~48–54% of retention cost</b> and, after <b>isotonic calibration</b>,
emits well-calibrated probabilities (ECE {u['ece']:.3f} → {i['ece']:.3f}).
Three further iterations (navigation, sequence-order, difficulty) were ruled out, and a learning
curve shows the model has plateaued — the signal is saturated, not data-starved.
</div>

<h2>1 · Problem &amp; target</h2>
<p>Predict whether a <i>new</i> user churns in their first day so the growth team can trigger a
retention treatment. <b>Target:</b> <code>churned = user_last_engagement &lt; first_engagement + 24h</code>
(positive class ≈ 23%). One row per <code>user_pseudo_id</code>; features use only events within the
24 h observation window (prediction time).</p>

<h2>2 · Methodology (what makes a claim defensible)</h2>
<ul>
<li><b>Primary metric:</b> PR-AUC (average precision) on the churn class — minority-class-appropriate.</li>
<li><b>Decision rule:</b> a candidate wins only if its <b>bootstrap CI does not overlap</b> the incumbent's
(point-estimate gains are insufficient). Selection on 5-fold OOF over train; frozen test scored once.</li>
<li><b>Leakage audit (H3):</b> every feature set checked — univariate directed AUC + ablation; tripwire at AUC ≥ 0.85.</li>
<li><b>Reproducibility:</b> hash-pinned data, fixed seeds, MLflow tracking, clean-venv rebuild matches within ±0.005.</li>
<li><b>Compute:</b> parallelism capped (no <code>n_jobs=-1</code>) to leave CPU for other processes.</li>
</ul>

<h2>3 · Model progression (the wins)</h2>
<table>
{row(["Feature set", "CV PR-AUC [95% CI]", "Test PR-AUC", "Test ROC-AUC", "Cost savings"], th=True)}
{''.join(row(r) for r in progression)}
</table>
<p>Each step cleared the non-overlapping-CI bar. <b>Feature engineering beat architecture search</b>:
the per-iteration lift (+0.05–0.08) exceeded the spread across XGBoost/LightGBM/HistGB/CatBoost, and
the biggest jump came from session-level features (<code>engagement_time_msec</code> + 30-min gap sessionization,
since this 2018 export has no <code>ga_session_id</code>).</p>

<h2>4 · Learning curve — is the ceiling data or signal?</h2>
<figure><img alt="learning curve" src="data:image/png;base64,{lc_img}">
<figcaption>v3 champion: test PR-AUC vs training-set size (5 stratified repeats per point).</figcaption></figure>
<p><b>Plateaued.</b> Going 80% → 100% of the data adds only <b>+0.006</b> PR-AUC (within noise), while the
train score stays pinned near 1.0 (train/test gap ≈ 0.40). The ~0.59 test ceiling is set by the problem's
inherent predictability, <b>not</b> by sample count — so more data (or a data-hungry sequence DNN) is
unlikely to help.</p>

<h2>5 · What was ruled out (honest negatives)</h2>
<table>
{row(["Feature set (tuned)", "CV PR-AUC [95% CI]", "Why it failed"], th=True)}
{''.join(row(r) for r in negatives)}
</table>
<p>All three overlap or sit below the v3 champion (<b>0.548 [0.520, 0.575]</b>) → <span class="bad">rejected</span>.
The non-overlapping-CI gate stopped plausible-but-null feature sets from being declared winners on tiny
point estimates. A learning-curve + Markov/n-gram analysis jointly closed the door on a sequence DNN.</p>

<h2>6 · Cost-based operating point (Q1 resolved)</h2>
<p>Stakeholder cost ratio: a false negative (missed churner) costs ~3–4× a false positive (wasted incentive).
The threshold is chosen by minimising <code>#FP·1 + #FN·c</code> on train OOF — not a precision floor
(the earlier precision ≥ 0.60 target was infeasible and is retired).</p>
<table>
{row(["C_FN/C_FP", "threshold", "precision", "recall", "savings vs best naive"], th=True)}
{''.join(row(r) for r in cost)}
</table>
<p>The model-driven policy beats both naive policies (treat-none / treat-all) and the default 0.5 — a
genuinely useful retention tool at ~88–96% recall.</p>

<h2>7 · Calibration of the champion</h2>
<figure><img alt="reliability diagram" src="data:image/png;base64,{cal_img}">
<figcaption>Reliability on the frozen test. The raw model is overconfident (<code>scale_pos_weight=4.1</code> inflates probabilities); calibration pulls it onto the diagonal.</figcaption></figure>
<table>
{row(["Variant", "Brier ↓", "ECE ↓", "PR-AUC", "ROC-AUC"], th=True)}
{row(["uncalibrated", f"{u['brier']:.4f}", f"{u['ece']:.4f}", f"{u['pr_auc']:.3f}", f"{u['roc_auc']:.3f}"])}
{row(["sigmoid (Platt)", f"{s['brier']:.4f}", f"{s['ece']:.4f}", f"{s['pr_auc']:.3f}", f"{s['roc_auc']:.3f}"])}
{row([f"<b>isotonic ★</b>", f"<b>{i['brier']:.4f}</b>", f"<b>{i['ece']:.4f}</b>", f"{i['pr_auc']:.3f}", f"{i['roc_auc']:.3f}"])}
</table>
<p><b>Isotonic calibration</b> more than halves ECE ({u['ece']:.3f} → {i['ece']:.3f}) and lowers Brier
({u['brier']:.3f} → {i['brier']:.3f}); because <code>CalibratedClassifierCV(cv=5)</code> averages 5 fold-models,
ranking even improves slightly (PR-AUC {u['pr_auc']:.3f} → {i['pr_auc']:.3f}). Calibrated probabilities make
treatment <b>prioritisation</b> meaningful. Saved to <code>champion_v3_calibrated_isotonic.joblib</code>.</p>

<h2>8 · Decision &amp; next steps</h2>
<ul>
<li><b>Adopt</b> XGBoost on the v3 (session) feature set, <b>isotonic-calibrated</b>, operated at the cost-optimal threshold.</li>
<li><b>Stop behavioral feature engineering</b> — three consecutive negatives + a plateaued learning curve show the in-app signal is saturated.</li>
<li><b>Remaining levers</b> are non-behavioral (acquisition source/campaign, device/network quality, first-session crash signals) or simply shipping v3.</li>
<li><b>Before production:</b> confirm the exact FN/FP cost &amp; daily treatment budget (Q2), then a shadow/canary rollout.</li>
</ul>

<footer>
Reproducibility: hash-pinned datasets, fixed seeds (42), MLflow runs, clean-venv rebuild within ±0.005, full pytest suite green.
Champion: XGBoost (v3 features) + isotonic calibration · CV PR-AUC 0.548 · test PR-AUC 0.591 · ROC-AUC 0.847 · Brier {i['brier']:.3f}.
Generated by <code>experiments/17_build_report.py</code>.
</footer>
</div></body></html>"""

OUT.write_text(html)
print(f"wrote {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    pass
