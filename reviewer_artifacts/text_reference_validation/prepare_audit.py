#!/usr/bin/env python3
"""Build deterministic, blinded offline interfaces for two independent authors."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments" / "reference_information"
PROTOCOL_VERSION = "text-reference-audit-v1"
QUESTIONS = [
    ("task_faithful", "Task faithful", "Is the reference consistent with the user task?"),
    ("criteria_consistent", "Criteria consistent", "Does it avoid contradicting or misrepresenting the stored criteria?"),
    ("unsupported_details_absent", "Free of unsupported details", "Are all material requirements supported by the task or stored criteria?"),
    ("observable_final_state", "Observable final state", "Does it describe evidence that could be judged from a final webpage screenshot?"),
    ("sufficiently_specific", "Sufficiently specific", "Is it specific enough to guide verification without inventing requirements?"),
    ("overall_usable", "Overall usable", "Is it suitable as an expected-outcome description for this verification setup?"),
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def blinded_order(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    ordered = records[:]
    rng.shuffle(ordered)
    # Deterministically avoid adjacent references from the same task.
    for i in range(1, len(ordered)):
        if ordered[i]["instance_id"] == ordered[i - 1]["instance_id"]:
            for j in range(i + 1, len(ordered)):
                if ordered[j]["instance_id"] not in {ordered[i - 1]["instance_id"], ordered[i]["instance_id"]}:
                    ordered[i], ordered[j] = ordered[j], ordered[i]
                    break
    assert all(ordered[i]["instance_id"] != ordered[i - 1]["instance_id"] for i in range(1, len(ordered)))
    return ordered


def build_html(author: str, records: list[dict], package_hash: str) -> str:
    public_records = [
        {
            "blind_id": r["blind_id"],
            "domain": r["domain"],
            "task_text": r["task_text"],
            "eval_criteria": r["eval_criteria"],
            "text_reference": r["text_reference"],
            "reference_sha256": r["reference_sha256"],
        }
        for r in records
    ]
    payload = json.dumps(public_records, ensure_ascii=False).replace("</", "<\\/")
    questions = json.dumps(QUESTIONS, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Text Reference Audit Author {author}</title>
<style>
:root{{--ink:#172033;--muted:#5b6475;--line:#d7dce5;--panel:#f7f8fb;--accent:#175cd3;--good:#067647;--warn:#b54708}}
*{{box-sizing:border-box}} body{{margin:0;font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:#eef1f6}}
header{{position:sticky;top:0;z-index:5;background:white;border-bottom:1px solid var(--line);padding:14px 24px;display:flex;gap:20px;align-items:center}}
h1{{font-size:20px;margin:0}} .progress{{margin-left:auto;min-width:280px}} progress{{width:100%;height:14px}} .small{{font-size:13px;color:var(--muted)}}
main{{max-width:1120px;margin:24px auto;padding:0 20px 80px}} .notice,.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px}}
.notice{{border-left:5px solid var(--accent)}} h2{{font-size:17px;margin:0 0 8px}} .source{{white-space:pre-wrap;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin-top:8px}}
.reference{{font-size:19px;line-height:1.55;background:#fffaf0;border:1px solid #f2d6a2;border-radius:8px;padding:16px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .question{{border:1px solid var(--line);border-radius:9px;padding:13px}} .question strong{{display:block}}
.choices{{display:flex;gap:7px;margin-top:10px}} button{{border:1px solid #aab2c0;background:white;border-radius:7px;padding:8px 13px;cursor:pointer;font-weight:600}}
button:hover{{border-color:var(--accent)}} button.selected{{background:var(--accent);border-color:var(--accent);color:white}} button.no.selected{{background:#b42318}} button.uncertain.selected{{background:var(--warn)}}
textarea{{width:100%;min-height:76px;border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit}}
.nav{{display:flex;gap:10px;align-items:center;position:sticky;bottom:0;background:white;border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:18px}} .nav button.primary{{background:var(--accent);color:white;border-color:var(--accent)}} .nav .spacer{{flex:1}}
.done{{color:var(--good);font-weight:700}} .incomplete{{color:var(--warn);font-weight:700}} label.confirm{{display:block;margin:18px 0}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}} header{{align-items:flex-start;flex-wrap:wrap}} .progress{{margin-left:0;width:100%}}}}
</style></head><body>
<header><div><h1>Blinded text-reference audit · Author {author}</h1><div class="small">Protocol {PROTOCOL_VERSION} · package {package_hash[:12]}</div></div><div class="progress"><div id="progressText" class="small"></div><progress id="progress" max="294" value="0"></progress></div></header>
<main>
<section class="notice"><strong>Independent review.</strong> Do not discuss labels or view the other author's export until both final exports are complete. Generation arm, verifier outputs and outcome labels are hidden. Stored criteria are shown only as audit evidence and may or may not have been given to the generator.</section>
<section class="card"><div class="small" id="position"></div><h2 id="recordTitle"></h2><div class="small" id="domain"></div><h2>User task</h2><div class="source" id="task"></div><h2>Stored benchmark criteria</h2><div class="source" id="criteria"></div><h2>Generated expected-outcome reference</h2><div class="reference" id="reference"></div></section>
<section class="card"><div class="grid" id="questions"></div><h2 style="margin-top:18px">Optional note</h2><textarea id="notes" placeholder="Explain No or Uncertain ratings, or record another material issue."></textarea></section>
<section class="card"><label class="confirm"><input type="checkbox" id="confirm"> I completed this audit independently without viewing the other author's labels or verifier outputs.</label><div id="finalStatus" class="small"></div></section>
<div class="nav"><button id="prev">Previous</button><button id="next" class="primary">Save and next</button><button id="jump">Next incomplete</button><span class="spacer"></span><button id="importBtn">Import backup</button><input type="file" id="importFile" accept="application/json" hidden><button id="backup">Export backup</button><button id="export">Export final JSON</button></div>
</main>
<script>
const AUTHOR={json.dumps(author)}, PACKAGE_HASH={json.dumps(package_hash)}, PROTOCOL={json.dumps(PROTOCOL_VERSION)};
const RECORDS={payload}; const QUESTIONS={questions}; const KEY=`text-ref-audit-${{AUTHOR}}-${{PACKAGE_HASH}}`;
let state=JSON.parse(localStorage.getItem(KEY)||'{{}}'); state.labels=state.labels||{{}}; state.index=state.index||0; state.confirmed=!!state.confirmed;
const $=id=>document.getElementById(id); const values=['yes','no','uncertain'];
function current(){{return RECORDS[state.index]}} function save(){{state.confirmed=$('confirm').checked;localStorage.setItem(KEY,JSON.stringify(state));updateProgress()}}
function complete(label){{return QUESTIONS.every(q=>values.includes(label?.[q[0]]))}}
function render(){{const r=current(),l=state.labels[r.blind_id]||{{}};$('position').textContent=`Record ${{state.index+1}} of ${{RECORDS.length}} · ${{r.blind_id}}`;$('recordTitle').textContent='Reference content review';$('domain').textContent=`Domain: ${{r.domain}}`;$('task').textContent=r.task_text;$('criteria').textContent=r.eval_criteria||'(No stored criteria text)';$('reference').textContent=r.text_reference;
 $('questions').innerHTML='';for(const [key,title,prompt] of QUESTIONS){{const box=document.createElement('div');box.className='question';box.innerHTML=`<strong>${{title}}</strong><span class="small">${{prompt}}</span><div class="choices"></div>`;const choices=box.querySelector('.choices');for(const v of values){{const b=document.createElement('button');b.type='button';b.textContent=v[0].toUpperCase()+v.slice(1);b.className=(v==='no'?'no ':v==='uncertain'?'uncertain ':'')+(l[key]===v?'selected':'');b.onclick=()=>{{state.labels[r.blind_id]=state.labels[r.blind_id]||{{}};state.labels[r.blind_id][key]=v;save();render()}};choices.appendChild(b)}}$('questions').appendChild(box)}}
 $('notes').value=l.notes||'';$('confirm').checked=state.confirmed;updateProgress()}}
function updateProgress(){{const n=RECORDS.filter(r=>complete(state.labels[r.blind_id])).length;$('progress').value=n;$('progressText').textContent=`${{n}} / ${{RECORDS.length}} complete`;$('finalStatus').innerHTML=n===RECORDS.length?(state.confirmed?'<span class="done">Ready for final export.</span>':'<span class="incomplete">Check the independence confirmation to enable final export.</span>'):`<span class="incomplete">${{RECORDS.length-n}} records remain incomplete.</span>`}}
$('notes').oninput=()=>{{const r=current();state.labels[r.blind_id]=state.labels[r.blind_id]||{{}};state.labels[r.blind_id].notes=$('notes').value;save()}};$('confirm').onchange=save;
$('prev').onclick=()=>{{save();state.index=Math.max(0,state.index-1);render();scrollTo(0,0)}};$('next').onclick=()=>{{save();state.index=Math.min(RECORDS.length-1,state.index+1);render();scrollTo(0,0)}};
$('jump').onclick=()=>{{save();let j=RECORDS.findIndex((r,i)=>i>state.index&&!complete(state.labels[r.blind_id]));if(j<0)j=RECORDS.findIndex(r=>!complete(state.labels[r.blind_id]));if(j>=0){{state.index=j;render();scrollTo(0,0)}}}};
function download(final){{save();const completed=RECORDS.filter(r=>complete(state.labels[r.blind_id])).length;if(final&&(completed!==RECORDS.length||!state.confirmed)){{alert('Complete all records and confirm independence before final export.');return}}const out={{protocol_version:PROTOCOL,package_hash:PACKAGE_HASH,author_code:AUTHOR,complete:final&&completed===RECORDS.length&&state.confirmed,independence_confirmed:state.confirmed,record_count:RECORDS.length,completed_count:completed,order:RECORDS.map(r=>r.blind_id),labels:state.labels,exported_utc:new Date().toISOString()}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)+'\\n'],{{type:'application/json'}}));a.download=`text_reference_audit_author_${{AUTHOR}}${{final?'':'_backup'}}.json`;a.click();URL.revokeObjectURL(a.href)}}
$('backup').onclick=()=>download(false);$('export').onclick=()=>download(true);$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{{const x=JSON.parse(await e.target.files[0].text());if(x.author_code!==AUTHOR||x.package_hash!==PACKAGE_HASH){{alert('Backup does not match this author/package.');return}}state.labels=x.labels||{{}};state.index=0;state.confirmed=!!x.independence_confirmed;save();render()}};
render();
</script></body></html>"""


def main() -> None:
    tasks = {r["instance_id"]: r for r in read_jsonl(EXP / "tasks.jsonl")}
    generations = read_jsonl(EXP / "generation_results.jsonl")
    assert len(generations) == 294
    assert len({r["job_id"] for r in generations}) == 294
    records = []
    mapping = {}
    for row in generations:
        assert row["status"] == "ok" and row["text_reference"].strip()
        task = tasks[row["instance_id"]]
        blind_id = "TR-" + hashlib.sha256((PROTOCOL_VERSION + "|" + row["job_id"]).encode()).hexdigest()[:12].upper()
        ref_hash = hashlib.sha256(row["text_reference"].encode()).hexdigest()
        record = {
            "blind_id": blind_id,
            "instance_id": row["instance_id"],
            "domain": row["instance_id"].split("_")[1],
            "task_text": task["task_text"],
            "eval_criteria": task.get("eval_criteria") or "",
            "text_reference": row["text_reference"],
            "reference_sha256": ref_hash,
        }
        records.append(record)
        mapping[blind_id] = {
            "job_id": row["job_id"],
            "instance_id": row["instance_id"],
            "variant": row["variant"],
            "domain": record["domain"],
            "reference_sha256": ref_hash,
        }
    assert len(mapping) == 294
    package_hash = hashlib.sha256(canonical(sorted(records, key=lambda r: r["blind_id"]))).hexdigest()
    (HERE / "blind_map.json").write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "package_hash": package_hash, "records": mapping}, indent=2) + "\n")
    orders = {"A": blinded_order(records, 2026092701), "B": blinded_order(records, 2026092702)}
    for author, order in orders.items():
        (HERE / f"author_{author.lower()}_review.html").write_text(build_html(author, order, package_hash))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "package_hash": package_hash,
        "record_count": 294,
        "task_count": 147,
        "questions": [q[0] for q in QUESTIONS],
        "author_orders": {a: [r["blind_id"] for r in o] for a, o in orders.items()},
        "source_files": {
            "tasks": "experiments/reference_information/tasks.jsonl",
            "generations": "experiments/reference_information/generation_results.jsonl",
        },
    }
    (HERE / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared 294 blinded records; package {package_hash}")
    print("Author orders are independently randomized and contain no adjacent task pairs.")


if __name__ == "__main__":
    main()
