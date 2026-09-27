"""Analyze all scheduled tasks, retaining invalid outcomes in operational results."""
import argparse,csv,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,chi2
P=Path(__file__).resolve().parent
attempts=[json.loads(l) for l in (P/'verification_results.jsonl').read_text().splitlines() if l.strip()]
rows=[r for r in attempts if r['status']=='ok']
ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
assert len({r['job_id'] for r in rows})==len(rows),'Duplicate successful job'
counts=Counter(r['model_label'] for r in rows)
complete_models={m for m,n in counts.items() if n==294}
if not args.allow_partial:assert len(rows)==1470,'Incomplete experiment'
all_successes=rows;rows=[r for r in rows if r['model_label'] in complete_models]
assert rows,'No completed models yet'
assert all(r['status']=='ok' for r in rows),'Unresolved request failures'
VALID={'SUCCESS','FAILURE'};rng=np.random.default_rng(20260927);out=[]
for m in sorted({r['model_label'] for r in rows}):
 arms={v:{r['instance_id']:r for r in rows if r['model_label']==m and r['variant']==v} for v in ['with_criteria','task_only']}
 a,b=arms['with_criteria'],arms['task_only'];assert len(a)==len(b)==147 and a.keys()==b.keys()
 ids=sorted(i for i in a if a[i].get('verdict') in VALID and b[i].get('verdict') in VALID)
 cells=[0]*4;delta=[]
 for i in ids:
  assert a[i]['ground_truth']==b[i]['ground_truth'] and a[i]['screenshot_sha256']==b[i]['screenshot_sha256']
  x=a[i]['verdict']==a[i]['ground_truth'];y=b[i]['verdict']==b[i]['ground_truth'];cells[0 if x and y else 1 if x else 2 if y else 3]+=1;delta.append(int(x)-int(y))
 both,bc,cc,neither=cells;discord=bc+cc
 p=1. if discord==0 else binomtest(bc,discord).pvalue if discord<25 else chi2.sf(max(0,abs(bc-cc)-1)**2/discord,1)
 d=np.array(delta);boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)*100;lo,hi=np.quantile(boot,[.025,.975])
 row=dict(model=m,n_joint=len(ids),both_correct=both,criteria_only_correct=bc,task_only_correct=cc,neither_correct=neither,criteria_accuracy=(both+bc)/len(ids),task_only_accuracy=(both+cc)/len(ids),difference_pp=d.mean()*100,ci95_low_pp=lo,ci95_high_pp=hi,p_raw=p,test='none' if discord==0 else 'exact' if discord<25 else 'asymptotic_cc')
 for v,arm in arms.items():
  valid=[r for r in arm.values() if r.get('verdict') in VALID];tp=sum(r['verdict']==r['ground_truth']=='SUCCESS' for r in valid);fp=sum(r['verdict']=='SUCCESS' and r['ground_truth']=='FAILURE' for r in valid);fn=sum(r['verdict']=='FAILURE' and r['ground_truth']=='SUCCESS' for r in valid)
  row[v+'_valid']=len(valid);row[v+'_all_task_accuracy']=sum(r['verdict']==r['ground_truth'] for r in valid)/147;row[v+'_success_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
 (P/f'matched_ids_{m}.txt').write_text('\n'.join(ids)+'\n');out.append(row)
ordered=sorted(out,key=lambda r:r['p_raw']);prev=0
for k,r in enumerate(ordered):r['p_bonferroni']=min(1,5*r['p_raw']);prev=max(prev,min(1,(5-k)*r['p_raw']));r['p_holm']=prev
with (P/'paired_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
summary={'verification_calls':len(attempts),'completed_verifications':len(all_successes),'analysis_status':'complete' if len(all_successes)==1470 else 'partial_completed_models_only','planned_multiplicity_family':5,'successes_per_model':dict(counts),'failed_attempts':len(attempts)-len(all_successes),'generation_calls':len((P/'generation_results.jsonl').read_text().splitlines()),'returned_models':dict(Counter(r.get('returned_model') for r in rows)),'cost_upper_estimate_usd':sum(r['cost_upper_estimate_usd'] for r in attempts)+sum(json.loads(l)['cost_upper_estimate_usd'] for l in (P/'generation_results.jsonl').read_text().splitlines()),'results':out}
(P/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
