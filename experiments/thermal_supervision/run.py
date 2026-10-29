"""Apply finite post-density CTG transactions to a calibrated native checkpoint.

No full trainer, densification or camera estimation is run by this command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .data import load_manifest
from .profiled import ProfiledTransferEngine
from .runtime import ThermalModel


def restore_rng(payload):
    if 'rng_cpu' in payload:
        torch.set_rng_state(payload['rng_cpu'].cpu())
        if payload.get('rng_cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in payload['rng_cuda']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    document, batches = load_manifest(manifest)
    model, payload = ThermalModel.load(Path(args.checkpoint))
    engine = ProfiledTransferEngine(model)
    for batch in batches:
        engine._preflight(*batch)
    # Fingerprint arrays too: resumed anchors cannot silently refer to new data.
    digest = hashlib.sha256(manifest.read_bytes())
    selected_ids = {o.frame.view_id for batch in batches for o in [batch[0], *batch[1], *batch[2]]}
    for row in sorted(document['observations'], key=lambda r: r['id']):
        if row['id'] in selected_ids:
            for field in ('target', 'weight'):
                digest.update((manifest.parent/row[field]).read_bytes())
    signature = digest.hexdigest()
    start = 0
    if payload.get('extra', {}).get('engine'):
        extra = payload['extra']
        if extra['manifest_sha256'] != signature:
            raise ValueError('resume data changed')
        engine.load_state_dict(extra['engine'])
        start = extra['next_transaction']
        restore_rng(payload)
    if args.audit_only:
        print(json.dumps({'audit': 'passed', 'transactions': len(batches), 'next_transaction': start,
                          'primitives': len(model.point_ids), 'renders': model.render_calls}))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(start, len(batches)):
        record = engine.step(*batches[index])
        records.append(record)
        temporary = out/'checkpoint.pending.pt'
        model.save(temporary, extra={'engine': engine.state_dict(), 'manifest_sha256': signature,
            'next_transaction': index+1, 'scope': 'fixed topology; source-only thermal profiling'})
        temporary.replace(out/'checkpoint.pt')
        (out/'transactions.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
