"""Resumable matched experiment. Explicit --execute required; credentials never logged."""
import argparse,ast,base64,concurrent.futures,datetime,hashlib,json,os,threading,time,urllib.request,urllib.error
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-repository'
RATES={'gpt-4.1-nano':(.1,.4),'gpt-4.1-mini':(.4,1.6),'gpt-4.1':(2,8),'claude-sonnet-4-6':(3,15),'gemini-3.6-flash':(.75,3.75)}
STOP=threading.Event(); GEMINI_LOCK=threading.Lock(); GEMINI_LAST=0.; GEMINI_INTERVAL=6.
LOCK=threading.Lock(); RESERVED=0.;SPENT=0.;CAP=20.
def read(name):
 p=HERE/name
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def append(name,row):
 with LOCK:
  with (HERE/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def parse(raw):
 tree=ast.parse((REPO/'llm_clients.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_parse_response')
 env={'json':json};exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_parser','exec'),env)
 result=env['_parse_response'](raw)
 if not isinstance(result,dict):return {'verdict':'PARSE_ERROR'}
 if result.get('verdict') not in ('SUCCESS','FAILURE'):result['verdict']='PARSE_ERROR'
 return result

def request(provider,model,system,user,image=None,generation=False):
 headers={'Content-Type':'application/json'}
 if provider=='openai':
  url='https://api.openai.com/v1/chat/completions';headers['Authorization']='Bearer '+KEYS['OPENAI_API_KEY']
  content=user if image is None else [{'type':'image_url','image_url':{'url':'data:image/png;base64,'+image,'detail':'high'}},{'type':'text','text':user}]
  messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':content}]
  payload={'model':model,'messages':messages,'temperature':0,'max_completion_tokens':200 if generation else 512}
 elif provider=='anthropic':
  url='https://api.anthropic.com/v1/messages';headers.update({'x-api-key':KEYS['ANTHROPIC_API_KEY'],'anthropic-version':'2023-06-01'})
  payload={'model':model,'system':system,'messages':[{'role':'user','content':[{'type':'image','source':{'type':'base64','media_type':'image/png','data':image}},{'type':'text','text':user}]}],'max_tokens':512,'temperature':0}
 else:
  url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent';headers['x-goog-api-key']=KEYS['GEMINI_API_KEY']
  payload={'system_instruction':{'parts':[{'text':system}]},'contents':[{'role':'user','parts':[{'inline_data':{'mime_type':'image/png','data':image}},{'text':user}]}],'generation_config':{'temperature':.01,'max_output_tokens':4096,'response_mime_type':'application/json'}}
 return url,headers,payload

def call(job,provider,model,system,user,image=None,generation=False,extra=None,retries_left=3):
 global SPENT,RESERVED,GEMINI_LAST
 if STOP.is_set():raise RuntimeError('Batch stopped after a failed call')
 if provider=='gemini':
  with GEMINI_LOCK:
   time.sleep(max(0,GEMINI_INTERVAL-(time.monotonic()-GEMINI_LAST)));GEMINI_LAST=time.monotonic()
 if STOP.is_set():raise RuntimeError('Batch stopped after a failed call')
 url,headers,payload=request(provider,model,system,user,image,generation)
 record=dict(job,started_utc=utc(),requested_model=model,provider=provider,system_prompt=system,user_prompt=user,request_sha256=sha(payload),settings={k:v for k,v in payload.items() if k in ('temperature','max_tokens','max_completion_tokens','generation_config')},**(extra or {}))
 with LOCK:
  if SPENT+RESERVED+.1>CAP:raise RuntimeError('Accounting stop threshold reached')
  RESERVED+=.1
 append('attempts.jsonl',{'job_id':job['job_id'],'stage':'generation' if generation else 'verification','started_utc':record['started_utc'],'request_sha256':record['request_sha256']})
 t=time.monotonic();cost=.1
 try:
  with urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers),timeout=120) as resp:
   data=json.load(resp);request_id=resp.headers.get('x-request-id') or resp.headers.get('request-id')
  if provider=='openai':
   choice=data['choices'][0];raw=choice['message'].get('content') or '';usage=data.get('usage',{});it=usage.get('prompt_tokens',0);ot=usage.get('completion_tokens',0);finish=choice.get('finish_reason');returned=data.get('model')
  elif provider=='anthropic':
   raw=''.join(c.get('text','') for c in data.get('content',[]) if c.get('type')=='text');usage=data.get('usage',{});it=usage.get('input_tokens',0)+usage.get('cache_read_input_tokens',0)+usage.get('cache_creation_input_tokens',0);ot=usage.get('output_tokens',0);finish=data.get('stop_reason');returned=data.get('model')
  else:
   candidate=(data.get('candidates') or [{}])[0];raw=''.join(c.get('text','') for c in candidate.get('content',{}).get('parts',[]) if not c.get('thought'));usage=data.get('usageMetadata',{});it=usage.get('promptTokenCount',0);ot=usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0);finish=candidate.get('finishReason');returned=data.get('modelVersion')
  rate=RATES[model];cost=(it*rate[0]+ot*rate[1])/1e6
  record.update(status='ok',returned_model=returned,request_id=request_id,response_id=data.get('id',data.get('responseId')),raw_response=data,raw_text=raw,finish_reason=finish,usage=usage,input_tokens=it,output_tokens=ot,cost_upper_estimate_usd=cost)
  if generation:
   record['text_reference']=raw.strip()
   if not raw.strip() or finish not in ('stop','end_turn','STOP'):record['status']='generation_incomplete'
  else:record.update(parse(raw))
 except urllib.error.HTTPError as e:
  message=e.read().decode('utf-8',errors='replace')[:8000]
  for secret in KEYS.values():
   if secret:message=message.replace(secret,'[REDACTED]')
  record.update(status='http_error',http_status=e.code,error_detail=message,cost_upper_estimate_usd=cost)
 except Exception as e:
  record.update(status='request_error',error_type=type(e).__name__,cost_upper_estimate_usd=cost)
 finally:
  record.update(finished_utc=utc(),latency_s=round(time.monotonic()-t,3))
  append('generation_results.jsonl' if generation else 'verification_results.jsonl',record)
  with LOCK:RESERVED-=.1;SPENT+=cost
 if record['status']=='http_error' and record.get('http_status') in (429,500,502,503,504) and retries_left>0:
  time.sleep(30*(4-retries_left))
  return call(job,provider,model,system,user,image,generation,extra,retries_left-1)
 if record['status']!='ok':
  STOP.set()
  raise RuntimeError(f"Call stopped: {job['job_id']} {record['status']} {record.get('http_status','')}")
 return record

def main():
 global KEYS,SPENT,GEMINI_INTERVAL
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['generation','verification'],required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--pilot',action='store_true');ap.add_argument('--models',nargs='+');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--retry-reviewed',action='store_true');ap.add_argument('--gemini-interval',type=float,default=6);args=ap.parse_args();GEMINI_INTERVAL=args.gemini_interval
 if not args.execute:raise SystemExit('No API calls: supply --execute to run')
 KEYS={}
 for line in (REPO/'.env').read_text().splitlines():
  s=line.strip()
  if s and not s.startswith('#') and '=' in s:k,v=s.split('=',1);KEYS[k.strip()]=v.strip().strip('\"\'')
 for k in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY']:assert KEYS.get(k),'Missing credential '+k
 previous=read('generation_results.jsonl')+read('verification_results.jsonl');SPENT=sum(r.get('cost_upper_estimate_usd',0) for r in previous)
 if not args.retry_reviewed and any(r['status']!='ok' for r in previous):raise SystemExit('Previous failed call requires review before resuming')
 if args.stage=='generation':
  done={r['job_id'] for r in read('generation_results.jsonl')};jobs=[j for j in read('generation_requests.jsonl') if j['job_id'] not in done]
  if args.pilot:jobs=jobs[:2]
  # Pairs remain sequential within a task; four independent task workers.
  pairs=[jobs[i:i+2] for i in range(0,len(jobs),2)]
  def work(pair):
   for job in pair:call(job,'openai','gpt-4.1-nano','',job['request']['prompt'],generation=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for n,_ in enumerate(pool.map(work,pairs),1):
    if n%10==0 or n==len(pairs):print(f'Generation pairs {n}/{len(pairs)}; accounting ${SPENT:.3f}',flush=True)
 else:
  refs={r['job_id']:r for r in read('generation_results.jsonl')};assert len(refs)==294 and all(r['status']=='ok' for r in refs.values())
  tasks={r['instance_id']:r for r in read('tasks.jsonl')};prompts=json.loads((HERE/'verification_prompt.json').read_text())
  for t in tasks.values():assert t['screenshot_sha256'] and hashlib.sha256(Path(t['screenshot_path']).read_bytes()).hexdigest()==t['screenshot_sha256']
  done={r['job_id'] for r in read('verification_results.jsonl') if r['status']=='ok'};jobs=[j for j in read('verification_schedule.jsonl') if j['job_id'] not in done and (not args.models or j['model_label'] in args.models)]
  groups={m:[j for j in jobs if j['model_label']==m] for m in sorted({j['model_label'] for j in jobs})}
  def work(group):
   if args.pilot:group=group[:2]
   for n,job in enumerate(group,1):
    t=tasks[job['instance_id']];ref=refs[job['reference_job_id']]['text_reference'];user=prompts['user'].format(task_text=t['task_text'],text_reference=ref)
    image=base64.b64encode(Path(t['screenshot_path']).read_bytes()).decode()
    call(job,job['provider'],job['requested_model'],prompts['system'],user,image=image,extra={'ground_truth':t['ground_truth'],'screenshot_sha256':t['screenshot_sha256'],'reference_sha256':hashlib.sha256(ref.encode()).hexdigest(),'text_reference':ref})
    if len(group)>2 and (n%20==0 or n==len(group)):print(f"{job['model_label']}: {n}/{len(group)}; accounting ${SPENT:.3f}",flush=True)
  # Interleave model task-pairs; each pair remains sequential.
  paired_groups={}
  for model,group in groups.items():
   by_task={}
   for job in group:by_task.setdefault(job['instance_id'],[]).append(job)
   paired_groups[model]=list(by_task.values())
  batches=[]
  for offset in range(max((len(g) for g in paired_groups.values()),default=0)):
   for group in paired_groups.values():
    if offset<len(group):batches.append(group[offset])
  if args.pilot:batches=batches[:len(groups)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
   for n,_ in enumerate(pool.map(work,batches),1):
    if n%25==0 or n==len(batches):print(f'Verification pairs {n}/{len(batches)}; accounting ${SPENT:.3f}',flush=True)
 print(f'Stage complete; cumulative accounting ${SPENT:.4f}',flush=True)
if __name__=='__main__':main()
