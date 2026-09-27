"""Offline integrity audit; safe while append-only result logs are idle."""
import hashlib,json
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent
def read(name):return [json.loads(l) for l in (P/name).read_text().splitlines() if l.strip()]
gen=read('generation_results.jsonl');requests={r['job_id']:r for r in read('generation_requests.jsonl')};tasks={r['instance_id']:r for r in read('tasks.jsonl')};schedule={r['job_id']:r for r in read('verification_schedule.jsonl')};prompts=json.loads((P/'verification_prompt.json').read_text());success={r['job_id']:r for r in gen if r['status']=='ok'}
assert len(gen)==len(success)==294
for job_id,r in success.items():
 assert r['user_prompt']==requests[job_id]['request']['prompt']
 assert r['requested_model']=='gpt-4.1-nano' and r['text_reference'] and r['finish_reason']=='stop'
for t in tasks.values():assert hashlib.sha256(Path(t['screenshot_path']).read_bytes()).hexdigest()==t['screenshot_sha256']
rows=read('verification_results.jsonl');seen=set()
for r in rows:
 assert r['job_id'] in schedule
 if r['status']!='ok':continue
 assert r['job_id'] not in seen;seen.add(r['job_id']);j=schedule[r['job_id']];t=tasks[j['instance_id']];ref=success[j['reference_job_id']]['text_reference']
 assert r['text_reference']==ref and r['reference_sha256']==hashlib.sha256(ref.encode()).hexdigest()
 assert r['screenshot_sha256']==t['screenshot_sha256'] and r['ground_truth']==t['ground_truth']
 assert r['system_prompt']==prompts['system'] and r['user_prompt']==prompts['user'].format(task_text=t['task_text'],text_reference=ref)
 assert r['requested_model']==j['requested_model']
report={'checks_passed':True,'generation_results':len(gen),'successful_verifications':len(seen),'scheduled_verifications':len(schedule),'completed_per_model':dict(Counter(r['model_label'] for r in rows if r['status']=='ok')),'invalid_verdicts':dict(Counter(r['model_label'] for r in rows if r['status']=='ok' and r.get('verdict') not in ['SUCCESS','FAILURE'])),'failed_http_or_request_attempts':sum(r['status']!='ok' for r in rows)}
(P/'integrity_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
