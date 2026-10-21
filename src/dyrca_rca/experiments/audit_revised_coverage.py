"""Audit missing R2 control strata against frozen predictions, without retraining.

Observed topology signatures support exploratory matching. Protocol-declared
decoys are not silently relabelled as independently verified healthy services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dyrca_rca.experiments.backfill_revised_source import frozen_stage, sha
from dyrca_rca.experiments.complete_revised_experiments import read_json, write_json, normalize_service, bind_cases_to_root, validate_fresh_output

PROJECT=Path(__file__).resolve().parents[3]
METHODS={'Change-only':'change','Node-only':'node','DY-RCA':'dy'}


def topology_signature(case):
    """Match exact aggregate observed changes, without service IDs/targets."""
    phases=[]
    for phase in ('pre','during'):
        data=read_json(case/'topology'/(phase+'.json'))
        state=defaultdict(lambda:{'endpoints':set(),'pods':set(),'versions':set()})
        for svc in data.get('services',[]):
            if isinstance(svc.get('endpoint_ips'),list):state[svc['name']]['endpoints'].update(svc['endpoint_ips'])
        for pod in data.get('pods',[]):
            if pod.get('service'):
                value=state[pod['service']];value['pods'].add(pod['name'])
                if pod.get('version') not in (None,'','unknown'):value['versions'].add(pod['version'])
        phases.append(state)
    before,after=phases
    changed=[]
    for service in sorted(set(before)|set(after)):
        a,b=before[service],after[service]
        vector=tuple(len(b[key]-a[key]) for key in ('endpoints','pods','versions'))+tuple(len(a[key]-b[key]) for key in ('endpoints','pods','versions'))
        if any(vector):changed.append(vector)
    return sorted(changed)


def declared_decoys(metadata):
    keys=['topology_target_service']
    if metadata.get('control_mode')=='multi':keys.append('topology_target_service_2')
    return sorted(set(metadata[key] for key in keys if metadata.get(key)))


def run(args):
    raw,out=args.data_root.resolve(),args.out_dir.resolve()
    validate_fresh_output(raw,out,'dyrca_rca.experiments.audit_revised_coverage')
    if args.seed!=42 or args.planner!='rule':raise ValueError('Frozen seed42/rule protocol required')
    index=frozen_stage(args.experiment_root,args.system,'01_index')
    ranking=frozen_stage(args.experiment_root,args.system,'ranking_s42_t42')
    audit=frozen_stage(args.experiment_root,args.system,'02_audit')
    if not all((index,ranking,audit)):raise ValueError('Completed collection/ranking receipts required')
    cases=index/'cases'
    with (audit/'collection_audit.jsonl').open() as stream:audits=[json.loads(line) for line in stream if line.strip()]
    frozen_hashes={row['case_id']:row['hashes'] for row in audits}
    source=ranking/'predictions.jsonl'
    predictions=defaultdict(dict)
    with source.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['method'] in METHODS and row['split']=='control':predictions[row['experiment']][row['method']]=row['prediction']
    # All predictions already existed and are receipt/hash-bound before label access here.
    selected=sorted(predictions)[:args.limit] if args.limit else sorted(predictions)
    bind_cases_to_root([cases/name for name in selected],raw)
    source_hashes={}
    for name in selected:
        for relative in ('metadata.json','topology/pre.json','topology/during.json','ground_truth.json'):
            path=cases/name/relative
            checksum=sha(path)
            if checksum!=frozen_hashes[name][relative]:raise ValueError('Control source differs from frozen collection audit: '+name+'/'+relative)
            source_hashes[str(path.resolve())]=checksum
    signature_rows=[]
    for name in selected:
        case=cases/name
        metadata=read_json(case/'metadata.json')
        signature_rows.append({'experiment':name,'signature':topology_signature(case),'is_benign':bool(metadata.get('is_benign')),
                               'control_mode':metadata.get('control_mode'),
                               'decoys':declared_decoys(metadata),'target':None if metadata.get('is_benign') else read_json(case/'ground_truth.json')['root_cause_service'],
                               'metadata_sha256':sha(case/'metadata.json'),'predictions':predictions[name]})
    by_pattern=defaultdict(list)
    for row in signature_rows:by_pattern[json.dumps(row['signature'])].append(row)
    matched=[];pairs=[]
    for pattern,rows in sorted(by_pattern.items()):
        faults=[r for r in rows if not r['is_benign']];benign=[r for r in rows if r['is_benign']]
        if faults and benign:
            matched.extend(faults)
            for f in faults:
                pairs.append({'fault_experiment':f['experiment'],'benign_experiment':benign[0]['experiment'],
                              'signature':json.loads(pattern),'selection':'lexicographic benign match; with replacement; exploratory'})
    groups={}
    for subset,rows in (('fault_benign',[r for r in signature_rows if not r['is_benign']]),('samechange',matched),
                        ('multi_change',[r for r in signature_rows if r['control_mode']=='multi']),
                        ('benign',[r for r in signature_rows if r['is_benign']])):
        summary={}
        for method,key in METHODS.items():
            eligible=[r for r in rows if method in r['predictions']]
            nominees=[r['predictions'][method]['ranked_node_ids'][0] for r in eligible]
            decoy_hits=[any(nominee==decoy or nominee.startswith(decoy+'-cf-') for decoy in row['decoys']) for nominee,row in zip(nominees,eligible)]
            summary[key]={'n':len(eligible),'hit1':statistics.mean(r['predictions'][method]['ranked_node_ids'][0]==r['target'] for r in eligible) if eligible and subset!='benign' else None,
                          'decoy_nominated':sum(decoy_hits),'decoy_nomination_rate':statistics.mean(decoy_hits) if decoy_hits else None}
        groups[subset]=summary
    primary=[r for r in audits if r['scenario_id'] in ('T1','T2','T3','T4')]
    unsupported={'independent_causal_references':0,'independent_human_adjudicators':0,
                 'causal_verified_n':None,'unrealized_topology_changes':sum(not r['verified'] for r in primary),
                 'causal_unresolved_screened_runs':sum(r['verified'] for r in audits),'other_topology_screen_exclusions':sum(not r['verified'] for r in audits if r['scenario_id'] not in ('T1','T2','T3','T4')),
                 'reason':'Raw topology/telemetry and controller targets are available, but no external causal reference or independent adjudication is recorded.'}
    report={'schema_version':'r2-coverage-v2','status':'bounded_smoke' if args.limit else 'completed','system':args.system,'seed':args.seed,'planner':args.planner,'limit':args.limit,
            'scope':'Exploratory observed-change matching and protocol-decoy nomination; not independently healthy-decoy error',
            'prediction_sha256':sha(source),'signature_definition':'Sorted vectors: endpoint/Pod/version additions then removals; IDs omitted, full cardinality signature retained',
            'control_cases':len(signature_rows),'matched_fault_cases':len(matched),'matched_pairs':pairs,'groups':groups,
            'causal_reference_coverage':unsupported,'runner_sha256':sha(Path(__file__)),
            'frozen_source_hashes_verified':len(source_hashes),'source_sha256':source_hashes,
            'frozen_collection_hash_source_sha256':sha(audit/'collection_audit.jsonl')}
    if any(sha(Path(path))!=checksum for path,checksum in source_hashes.items()):
        raise ValueError('Control inputs changed during the coverage audit')
    out.mkdir(parents=True,exist_ok=True)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'control_source_records.json',signature_rows)
    write_json(out/'coverage_summary.json',report)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=PROJECT/'data/experiments_tosem_four_copies')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--experiment-root',type=Path,default=PROJECT/'artifacts/tosem_four_copies/experiments_20260930')
    p.add_argument('--system',choices=('boutique','trainticket'),default='boutique')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--planner',choices=('rule',),default='rule')
    p.add_argument('--limit',type=int)
    return p


if __name__=='__main__':
    summary=run(parser().parse_args())
    print(json.dumps({k:v for k,v in summary.items() if k not in ('matched_pairs','groups')},sort_keys=True))
