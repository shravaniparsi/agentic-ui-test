"""Export only explicitly approved, hash-verified reference bytes to a new set."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent


def export(decisions_path, destination):
    decisions = json.loads(Path(decisions_path).read_text())
    destination = Path(destination)
    exported = []
    prepared = []
    seen = set()
    for d in decisions:
        if d.get('status') != 'approved':
            continue
        required = ['target_instance_id', 'source_instance_id', 'frame_sha256', 'identity_evidence', 'completion_evidence', 'reviewer']
        if not all(d.get(k) for k in required):
            raise ValueError('Incomplete approval record')
        if not d.get('identity_verified') or not d.get('completion_supported'):
            raise ValueError('Approval lacks both validation gates')
        iid = d['target_instance_id']
        if iid in seen or not re.fullmatch(r'vwa_(classifieds|reddit|shopping)_\d+',iid):
            raise ValueError('Duplicate or invalid target ID')
        seen.add(iid)
        if not re.fullmatch(r'vwa_(classifieds|reddit|shopping)_\d+',d['source_instance_id']):
            raise ValueError('Invalid source ID')
        if iid != d['source_instance_id'] and not d.get('remapping_evidence'):
            raise ValueError('Remapping requires explicit evidence')
        source = json.loads((ROOT/'records'/f"{d['source_instance_id']}.json").read_text())
        frame = next((f for f in source['final_frames']+source.get('candidate_frames',[]) if f['sha256'] == d['frame_sha256']), None)
        if frame is None:
            raise ValueError('Approved frame not in audited source')
        src = (AUDIT/frame['path']).resolve()
        if not src.exists():
            # The public review package omits the large frame cache. For an
            # already-approved decision, use the released native frame after
            # checking the same recorded byte hash below.
            candidates = list((ROOT/'approved_references').glob(iid+'.*'))
            if len(candidates) != 1:
                raise ValueError('Approved source frame unavailable')
            src = candidates[0].resolve()
            if not src.is_relative_to((ROOT/'approved_references').resolve()):
                raise ValueError('Released frame path escapes approved directory')
        elif not src.is_relative_to((ROOT/'frames').resolve()):
            raise ValueError('Frame path escapes audited directory')
        b = src.read_bytes()
        if hashlib.sha256(b).hexdigest() != d['frame_sha256']:
            raise ValueError('Frame hash mismatch')
        target = destination/(iid+src.suffix)
        if target.exists() and target.read_bytes() != b:
            raise ValueError('Refusing to overwrite a different reference')
        prepared.append((target,b))
        exported.append(dict(d, reference_screenshot_path=str(target.resolve()), archive_sha256=source['archive_sha256'], source_file_id=source['source_file_id'], frame_resource=frame['resource'], frame_page_id=frame['page_id'], frame_source_path=frame['path'], frame_timestamp=frame['timestamp'], frame_dimensions=frame['dimensions'], frame_format=frame['format']))
    # Validate the entire batch before writing. Stale references must not survive a narrower manifest.
    expected={p.name for p,b in prepared}|{'manifest.json','task_ids.txt'}
    if destination.exists() and any(p.name not in expected for p in destination.iterdir()):
        raise ValueError('Destination contains stale or unrelated files; choose a fresh directory')
    destination.mkdir(parents=True,exist_ok=True)
    for target,b in prepared:
        target.write_bytes(b)
    (destination/'manifest.json').write_text(json.dumps(exported,indent=2)+'\n')
    (destination/'task_ids.txt').write_text(''.join(d['target_instance_id']+'\n' for d in exported))
    return exported


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', default=str(ROOT/'review_decisions.json'))
    parser.add_argument('--destination', default=str(ROOT/'approved_references'))
    args = parser.parse_args()
    print('Exported approved references:', len(export(args.decisions,args.destination)))
