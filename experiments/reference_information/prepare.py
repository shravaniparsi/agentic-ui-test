"""Prepare deterministic matched inputs; never makes API calls."""
import ast, hashlib, json, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'research-repository'
OUT=Path(__file__).resolve().parent

def literal(path,name):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def write(name,rows):
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

ids=sorted(set((REPO/'data/visual_subset_ids.txt').read_text().split()))
assert len(ids)==147
rows=[json.loads(x) for x in (REPO/'data/verification_dataset.jsonl').read_text().splitlines() if x.strip()]
assert len({r['instance_id'] for r in rows})==len(rows)
data={r['instance_id']:r for r in rows}
models=literal(REPO/'config.py','MODELS')
prompts=literal(REPO/'prompts/verification_prompts.py','PROMPTS')['B']
# Identical instruction wrapper; only the presence of the criteria field changes.
wrapper=('You are helping evaluate web automation agents. Given the task information below, '
         'produce a 1-3 sentence natural-language description of what the final web page '
         'should look like if the task was completed successfully. Focus on what semantic '
         'content / state should be visible. Do NOT describe the layout (colors, fonts, '
         "positions). Do NOT include 'In conclusion' or preamble. Output ONLY the description.\n\n")
rng=random.Random(20260927); jobs=[]; tasks=[]; verification=[]
for i in ids:
    r=data[i]; assert r['task_text'] and r['eval_criteria']
    path=REPO/r['final_screenshot_path']
    tasks.append(dict(instance_id=i,task_text=r['task_text'],eval_criteria=r['eval_criteria'],ground_truth=r['ground_truth'],screenshot_path=str(path),screenshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None))
    variants=['task_only','with_criteria'];rng.shuffle(variants)
    for v in variants:
        prompt=wrapper+'Task: '+r['task_text']+'\n\n'
        if v=='with_criteria':prompt+='Evaluation criteria: '+r['eval_criteria']+'\n\n'
        prompt+='Expected outcome description:'
        request=dict(model='gpt-4.1-nano',prompt=prompt,temperature=0,max_output_tokens=200)
        jobs.append(dict(job_id=f'{i}:{v}',instance_id=i,variant=v,request=request,request_sha256=digest(request)))
for m,spec in sorted(models.items()):
    order=list(ids);rng.shuffle(order)
    for i in order:
        variants=['task_only','with_criteria'];rng.shuffle(variants)
        for v in variants:
            verification.append(dict(job_id=f'{m}:{i}:{v}',model_label=m,requested_model=spec['model_id'],provider=spec['provider'],instance_id=i,variant=v,reference_job_id=f'{i}:{v}'))
assert len(jobs)==294 and len(verification)==1470
write('tasks.jsonl',tasks);write('generation_requests.jsonl',jobs);write('verification_schedule.jsonl',verification)
(OUT/'verification_prompt.json').write_text(json.dumps(prompts,indent=2)+'\n')
summary=dict(task_count=len(tasks),generation_calls=len(jobs),verification_calls=len(verification),screenshots_present=sum(r['screenshot_sha256'] is not None for r in tasks),generation_manifest_sha256=digest(jobs),seed=20260927,models={m:s['model_id'] for m,s in models.items()},status='prepared_not_executed')
(OUT/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
