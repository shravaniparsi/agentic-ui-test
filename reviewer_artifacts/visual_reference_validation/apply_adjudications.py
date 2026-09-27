"""Apply explicitly recorded visual judgments, without using model outcomes."""
import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
queue={q['instance_id']:q for q in json.loads((ROOT/'review_queue.json').read_text())}
decisions={d['target_instance_id']:d for d in json.loads((ROOT/'review_decisions.json').read_text())}
notes=json.loads((ROOT/'visual_adjudications.json').read_text())
repairs={
 'vwa_classifieds_170':('vwa_classifieds_170__candidate135.jpg','Samsung Star Wars phone listing shows Kylo Ren packaging, matching the inspected task image, with $1300 price and exact target URL 36313.'),
 'vwa_classifieds_21':('vwa_classifieds_21__candidate124.jpg','Dark Zora B the Bus is visible with listing title and exact target URL 33164.'),
 'vwa_shopping_11':('vwa_shopping_11__candidate355.jpg','Recorded target product page shows the Skinny Cow round-cookie sandwich in its manufacturer illustration; exact accepted product URL corroborates identity. This is an illustration lower on the product page, not the primary product photo.'),
 'vwa_shopping_27':('vwa_shopping_27__candidate110.jpg','Exact accepted white-table title and white three-legged table diagram are visible on target product page. The diagram is partly below viewport; it is not a complete product photograph.'),
}
for iid,(status,evidence) in notes.items():
    q=queue[iid]; f=q['frames'][-1]; d=decisions[iid]
    d.update(status=status,identity_verified=status=='approved',completion_supported=status=='approved',reviewer='Codex AI visual review (not independent human annotation)',review_level='visual_and_task_criteria' if status=='approved' else 'visual_review_unresolved',frame_sha256=f['sha256'],completion_evidence=evidence,configuration_sha256=q['configuration_sha256'],selection_rationale='Unmodified global last recorded screencast frame; task-specific content adjudicated against published criteria.',validation_scope='Screening for benchmark correspondence and visible evidence; not an independent human validation or fresh benchmark evaluator execution.')
    d['identity_evidence']=evidence if status=='approved' else 'Unresolved aspects are described in completion_evidence; filename alone is not accepted.'
    d['task_definition']=dict(task=q['task'],start_url=q['start_url'],evaluation=q['evaluation'])
    d['recorded_page_context']=f.get('last_snapshot')
for iid,(filename,evidence) in repairs.items():
    r=json.loads((ROOT/'records'/f'{iid}.json').read_text()); q=queue[iid]
    f=next(f for f in r['candidate_frames'] if Path(f['path']).name==filename)
    snaps=[s for s in r['snapshots'] if s['pageId']==f['page_id'] and s['timestamp']<=f['timestamp']]
    d=decisions[iid]
    d.update(status='approved',identity_verified=True,completion_supported=True,reviewer='Codex AI visual review (not independent human annotation)',review_level='recovered_recorded_frame',frame_sha256=f['sha256'],identity_evidence=evidence,completion_evidence=evidence,configuration_sha256=q['configuration_sha256'],recorded_page_context=snaps[-1],selection_rationale='Earlier unmodified recorded frame on the accepted target page, after arrival and before scrolling to less informative content. This curated-frame rule differs from historical global-last-frame extraction.',validation_scope='Benchmark-target correspondence and visible content; viewport completeness is not guaranteed.',task_definition=dict(task=q['task'],start_url=q['start_url'],evaluation=q['evaluation']))
for iid,d in decisions.items():
    d['eligible_for_screened_set']=d['status']=='approved'
    d['task_definition']=dict(task=queue[iid]['task'],start_url=queue[iid]['start_url'],evaluation=queue[iid]['evaluation'])
    if d['status']=='approved':
        d['acquisition_caveat']='Native recorded encoding/viewport/recorder overlays retained. Identity screening does not equalize acquisition with agent screenshots.'
(ROOT/'review_decisions.json').write_text(json.dumps(list(decisions.values()),indent=2)+'\n')
print(dict(Counter(d['status'] for d in decisions.values())))
