"""Prepare a separate dataset from an explicitly approved, hash-verified manifest."""
import argparse
import hashlib
import json
from pathlib import Path

def prepare(dataset, manifest, output):
    dataset, manifest, output=map(Path,(dataset,manifest,output))
    entries=json.loads(manifest.read_text())
    if not entries:
        raise ValueError('Empty reference manifest')
    approved={}
    for e in entries:
        iid=e['target_instance_id']
        if iid in approved:
            raise ValueError('Duplicate manifest ID')
        required=['identity_evidence','completion_evidence','reviewer','archive_sha256','frame_sha256','selection_rationale']
        if e.get('status')!='approved' or not e.get('identity_verified') or not e.get('completion_supported') or not all(e.get(k) for k in required):
            raise ValueError('Unapproved or incomplete reference')
        if iid!=e.get('source_instance_id') and not e.get('remapping_evidence'):
            raise ValueError('Task remapping lacks evidence')
        path=Path(e['reference_screenshot_path'])
        if not path.is_absolute():path=manifest.parent/path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=e['frame_sha256']:
            raise ValueError('Reference image hash mismatch')
        approved[iid]=(e,path.resolve())
    rows=[];seen=set()
    for line in dataset.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);iid=row['instance_id']
        if iid not in approved:continue
        if iid in seen:raise ValueError('Duplicate dataset ID')
        seen.add(iid);e,path=approved[iid]
        row['reference_screenshot_path']=str(path)
        row['reference_validation']=dict(status='approved',frame_sha256=e['frame_sha256'],reviewer=e['reviewer'],source_instance_id=e['source_instance_id'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        rows.append(row)
    if seen!=set(approved):raise ValueError('Manifest IDs missing from dataset')
    # All checks precede the exclusive write; preserve existing datasets.
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    return len(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print('Prepared approved tasks:',prepare(a.dataset,a.manifest,a.output))
