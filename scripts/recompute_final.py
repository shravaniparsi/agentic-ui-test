import json, csv, math, os
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import binomtest, chi2 as chi2dist

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)

ALL_MODELS=['gpt-4.1-nano','gpt-4.1-mini','gpt-4.1','claude-sonnet-4','gemini-3.6-flash']
# Only analyse models whose four condition files are all present. gemini-2.5-flash
# was retired by Google and its original run artifacts were not preserved, so it
# is absent unless it is re-run; every count below reflects the models actually
# included rather than a hard-coded 5.
MODELS=[m for m in ALL_MODELS
        if all(os.path.exists(f'{m}_{c}.jsonl') for c in ['A','B','C','D'])]
MISSING=[m for m in ALL_MODELS if m not in MODELS]
COND=['A','B','C','D']
PAIRS=[('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]
SUB=set(l.strip() for l in open('data/visual_subset_ids.txt') if l.strip())
assert len(SUB)==147
VALID={'SUCCESS','FAILURE'}

def load(path, subset=False):
    d={}
    for l in open(path):
        if not l.strip(): continue
        r=json.loads(l)
        if subset and r['instance_id'] not in SUB: continue
        d[r['instance_id']]=r
    return d

def mcnemar(b,c):
    n=b+c
    if n==0: return ('none',0.0,1.0)
    if n<25:
        p=binomtest(min(b,c), n, 0.5).pvalue
        return ('exact', float('nan'), p)
    chi=(abs(b-c)-1)**2/n
    p=1-chi2dist.cdf(chi,1)
    return ('asymptotic_cc', chi, p)

def metrics(rows):
    tp=fp=tn=fn=0
    for r in rows:
        v,g=r['verdict'],r['ground_truth']
        if v not in VALID: continue
        if g=='SUCCESS' and v=='SUCCESS': tp+=1
        elif g=='FAILURE' and v=='SUCCESS': fp+=1
        elif g=='FAILURE' and v=='FAILURE': tn+=1
        else: fn+=1
    n=tp+fp+tn+fn
    acc=(tp+tn)/n if n else float('nan')
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec=tp/(tp+fn) if tp+fn else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    fpr=fp/(fp+tn) if fp+tn else float('nan')
    return dict(n=n,tp=tp,fp=fp,tn=tn,fn=fn,acc=acc,prec=prec,rec=rec,f1=f1,fpr=fpr)

print(f'models analysed ({len(MODELS)}): ' + ', '.join(MODELS))
if MISSING:
    print(f'MODELS EXCLUDED (result files absent): ' + ', '.join(MISSING))
    print('  -> S1/T1/S2 counts below cover the included models only.')

# ---------- load all condition files (C/D restricted to subset) ----------
R={}
for m in MODELS:
    for c in COND:
        R[(m,c)]=load(f'{m}_{c}.jsonl', subset=(c in 'CD'))

# ---------- Table 1 check vs current files ----------
t1=[]
for m in MODELS:
    for c in COND:
        rows=list(R[(m,c)].values())
        errs=Counter(r['verdict'] for r in rows if r['verdict'] not in VALID)
        mt=metrics(rows)
        t1.append(dict(model=m,condition=c,total=len(rows),api_errors=errs.get('API_ERROR',0),parse_errors=errs.get('PARSE_ERROR',0),**mt))
with open(OUT/'T1_and_S3_error_rates.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=t1[0].keys()); w.writeheader(); w.writerows(t1)

# ---------- S1: 30 paired tests on jointly valid ----------
s1=[]
for m in MODELS:
    for a,b_ in PAIRS:
        A,B=R[(m,a)],R[(m,b_)]
        ids=[i for i in A if i in B and A[i]['verdict'] in VALID and B[i]['verdict'] in VALID]
        if a in 'CD' or b_ in 'CD': ids=[i for i in ids if i in SUB]
        b=c=both=neither=0
        for i in ids:
            ac=A[i]['verdict']==A[i]['ground_truth']; bc=B[i]['verdict']==B[i]['ground_truth']
            if ac and not bc: b+=1
            elif bc and not ac: c+=1
            elif ac and bc: both+=1
            else: neither+=1
        t,chi,p=mcnemar(b,c)
        idsdir=OUT/'S1_task_ids'; idsdir.mkdir(parents=True, exist_ok=True)
        with open(idsdir/f'{m}_{a}{b_}.txt','w') as f:
            f.write('\n'.join(sorted(ids))+'\n')
        s1.append(dict(model=m,pair=f'{a}->{b_}',n_joint=len(ids),both_correct=both,only_first_correct_b=b,only_second_correct_c=c,neither=neither,test=t,chi2=('' if math.isnan(chi) else round(chi,3)),p_raw=p))
NTESTS=len(s1)
alpha=0.05/NTESTS
ps=sorted([(r['p_raw'],k) for k,r in enumerate(s1)])
holm={}
for rank,(p,k) in enumerate(ps):
    holm[k]= p*(NTESTS-rank) <= 0.05 and all(ps[j][0]*(NTESTS-j)<=0.05 for j in range(rank+1))
for k,r in enumerate(s1):
    r['bonferroni_sig']=r['p_raw']<alpha; r['holm_sig']=holm[k]
with open(OUT/'S1_paired_tests.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=s1[0].keys()); w.writeheader(); w.writerows(s1)
sig=[r for r in s1 if r['bonferroni_sig']]
print(f'S1: {len(sig)} of {NTESTS} Bonferroni-significant; Holm: {sum(r["holm_sig"] for r in s1)}')
for r in sig: print('   ',r['model'],r['pair'])
print('Nano rows:'); [print('   ',r['pair'],'n=',r['n_joint'],'b=',r['only_first_correct_b'],'c=',r['only_second_correct_c'],r['test'],'chi2=',r['chi2'],'p=%.4g'%r['p_raw']) for r in s1 if r['model']=='gpt-4.1-nano']

# ---------- Cross-generator and no-criteria on 147 subset, jointly valid with A ----------
def crossgen(tag, outname):
    out=[]
    for m in MODELS:
        A=R[(m,'A')]; Bs=R[(m,'B')]
        xpath=f'results/cross_refs/{m}_B_{tag}.jsonl'
        if not os.path.exists(xpath):
            print(f'   {m}: no {tag} cross-ref file, skipped')
            continue
        X=load(xpath, subset=True)
        ids=[i for i in SUB if i in A and i in X and A[i]['verdict'] in VALID and X[i]['verdict'] in VALID]
        # A, B_self, B_x all on the SAME ids (also require B_self valid for the self column)
        ids_self=[i for i in ids if i in Bs and Bs[i]['verdict'] in VALID]
        accA=sum(A[i]['verdict']==A[i]['ground_truth'] for i in ids)/len(ids)
        accX=sum(X[i]['verdict']==X[i]['ground_truth'] for i in ids)/len(ids)
        accA_s=sum(A[i]['verdict']==A[i]['ground_truth'] for i in ids_self)/len(ids_self)
        accB_s=sum(Bs[i]['verdict']==Bs[i]['ground_truth'] for i in ids_self)/len(ids_self)
        b=c=0
        for i in ids:
            ac=A[i]['verdict']==A[i]['ground_truth']; xc=X[i]['verdict']==X[i]['ground_truth']
            if ac and not xc: b+=1
            elif xc and not ac: c+=1
        t,chi,p=mcnemar(b,c)
        mx=metrics([X[i] for i in ids])
        out.append(dict(model=m,n_joint=len(ids),A_acc=round(accA,3),B_x_acc=round(accX,3),lift_pp=round(100*(accX-accA),1),B_x_f1=round(mx['f1'],3),b=b,c=c,test=t,chi2=('' if math.isnan(chi) else round(chi,2)),p_raw=round(p,4),sig_05=p<0.05,sig_bonf5=p<0.01,
                        n_self=len(ids_self),A_acc_selfset=round(accA_s,3),B_self_acc=round(accB_s,3),lift_self_pp=round(100*(accB_s-accA_s),1)))
    with open(OUT/f'{outname}.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print(f'\n{outname}:')
    for r in out: print('   %-17s n=%3d  A=%.3f  Bx=%.3f  lift=%+5.1f  b=%2d c=%2d  %s p=%.4f  | self-ref lift on same subset=%+5.1f (n=%d)'%(r['model'],r['n_joint'],r['A_acc'],r['B_x_acc'],r['lift_pp'],r['b'],r['c'],r['test'],r['p_raw'],r['lift_self_pp'],r['n_self']))
crossgen('xref-claude','S2_cross_generator_claude')
crossgen('nocriteria','S2b_nocriteria')

# ---------- Table 4 Nano-C calibration on subset ----------
def ece(rows,bins,equal_mass=True):
    xs=[(r['confidence']/10.0, 1.0 if r['verdict']==r['ground_truth'] else 0.0) for r in rows if r['verdict'] in VALID and r.get('confidence') is not None]
    xs.sort()
    n=len(xs)
    if equal_mass:
        edges=[int(round(k*n/bins)) for k in range(bins+1)]
        chunks=[xs[edges[k]:edges[k+1]] for k in range(bins)]
    else: # 3 fixed bins 1-3,4-6,7-10
        chunks=[[x for x in xs if x[0]<=0.3],[x for x in xs if 0.3<x[0]<=0.6],[x for x in xs if x[0]>0.6]]
    e=0
    for ch in chunks:
        if not ch: continue
        conf=sum(a for a,_ in ch)/len(ch); acc=sum(b for _,b in ch)/len(ch)
        e+=len(ch)/n*abs(conf-acc)
    brier=sum((a-b)**2 for a,b in xs)/n
    return e,brier,n
rows=list(R[('gpt-4.1-nano','C')].values()) if ('gpt-4.1-nano','C') in R else []
e3,br,n=ece(rows,3,False); e15,_,_=ece(rows,15,True)
print(f'\nTable 4 Nano-C on 147 subset: n={n} ECE_3={e3:.3f} ECE_15={e15:.3f} Brier={br:.3f}  (paper had .285/.306/.317 on n=897)')
# sanity: recompute another cell to confirm method matches paper
rows=list(R[('claude-sonnet-4','A')].values()); e3,br,n=ece(rows,3,False); e15,_,_=ece(rows,15,True)
print(f'sanity Claude-A: n={n} ECE_3={e3:.3f} ECE_15={e15:.3f} Brier={br:.3f}  (paper .015/.030/.163)')

# ---------- precision / valid-rate summary for the paper ----------
print('\nSuccess-precision (TP/(TP+FP)) and valid rate, Condition B:')
for r in t1:
    if r['condition']=='B': print('   %-17s valid=%d/%d (%.1f%%) prec=%.3f fpr=%.3f'%(r['model'],r['n'],r['total'],100*r['n']/r['total'],r['prec'],r['fpr']))
print('Claude C valid: %d/147'%[r for r in t1 if r['model']=='claude-sonnet-4' and r['condition']=='C'][0]['n'])
