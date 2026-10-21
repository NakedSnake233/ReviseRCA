from copy import deepcopy

import pytest
import torch

from test_interaction import experiment, tmp_path, observe
from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.experiments.train import create_model
from dyrca_rca.interaction.temporal import PROTOCOL
from dyrca_rca.experiments.complete_revised_experiments import (
    corrupt_endpoints, fidelity, raw_objects, view_objects, paired_revisions,
    score_revisions, tensor_signature, version_groups,
    deployment_inventory,
    read_frozen_target, bind_cases_to_root, source_audit,
)


def test_defect_policy_ignores_labels_and_preserves_original(experiment):
    view=observe(experiment)
    before=deepcopy(view)
    broken,service,objects=corrupt_endpoints(view,42)
    assert view==before
    assert len(objects)==2 and {o['phase'] for o in objects}=={'pre','during'}
    # Only b has nonempty endpoints in both phases in this fixture.
    assert service=='b'
    assert len(set(o['service'] for o in objects))==1
    assert corrupt_endpoints(view,42)==(broken,service,objects)
    (experiment/'ground_truth.json').write_text('{"root_cause_service":"CHANGED_SECRET"}')
    assert corrupt_endpoints(observe(experiment),42)==(broken,service,objects)


def test_reference_excludes_future_and_reads_raw_values(experiment):
    view=observe(experiment)
    reference,availability,sources,unknown=raw_objects(experiment,view['as_of'])
    assert ('during','endpoints','a') in reference and reference[('during','endpoints','a')]==()
    assert set(sources)=={'pre','during'}
    assert view_objects(view)==(reference,availability)
    pre_only=raw_objects(experiment,'2026-01-01T00:00:00Z')[0]
    assert all(key[0]=='pre' for key in pre_only)


def test_zero_denominators_and_partial_repair_are_not_perfect_scores():
    ref={'a':1,'b':2,'c':3}
    assert fidelity(ref,ref)['repair'] is None
    assert fidelity({}, {})['fidelity'] is None
    measured=fidelity(ref,{'a':1,'b':0,'c':3},['a','b'])
    assert measured['repair']==50 and measured['damage']==0
    assert measured['fidelity']==pytest.approx(200/3)


def test_all_six_conditions_restore_half_or_all_without_training(experiment):
    torch.manual_seed(42);torch.set_num_threads(1)
    names,edges=feature_names(PROTOCOL)
    model=create_model('weighted_graphsage',len(names),len(edges))
    model.feature_protocol=PROTOCOL
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    view=observe(experiment)
    ref=raw_objects(experiment,view['as_of'])[0]
    records,histories,audit=paired_revisions(experiment,view,ref,42,model,'weighted_graphsage')
    assert len(records)==12 and len(histories)==12
    seeded={row['condition']:row for row in records if row['stratum']=='seeded'}
    assert seeded['correct']['content']['fidelity']==100
    assert seeded['correct']['content']['repair']==100
    assert seeded['partial']['content']['repair']==50
    assert seeded['incorrect']['content']['repair']==0
    assert seeded['incorrect']['content']['damage']==0
    assert seeded['redundant']['input_hash']==seeded['none']['input_hash']
    assert all(row['duplicate_state_equal'] for row in audit)
    assert all(torch.equal(frozen[k],v) for k,v in model.state_dict().items())
    # Missing targets stay in the denominator and contribute MRR zero.
    summary=score_revisions(records,{experiment.name:'ABSENT'})
    assert summary['seeded']['correct']['h1']==0
    assert summary['seeded']['correct']['mrr']==0
    assert seeded['correct']['tag_masked_mrr']==0
    assert seeded['nomination']['prediction']==seeded['none']['prediction']
    assert seeded['nomination']['recommendation_order'][0]==seeded['nomination']['nominated_service']
    assert seeded['nomination']['recommendation_policy'].startswith('forced post-processing')


def test_tensor_signature_covers_message_edges_and_all_features():
    item={'node_ids':['a'],'x':torch.zeros(1,2),'edge_index':torch.zeros(2,1,dtype=torch.long),'edge_attr':torch.zeros(1,3)}
    signature=tensor_signature(item)
    changed=deepcopy(item);changed['edge_attr'][0,2]=1
    assert tensor_signature(changed)!=signature
    changed=deepcopy(item);changed['node_ids']=['b']
    assert tensor_signature(changed)!=signature


def test_version_identity_does_not_invent_numeric_route_weights():
    groups=version_groups({'pods':[{'service':'checkout','version':'v1'},
                                    {'service':'checkout-v2','version':'v2'},
                                    {'service':'unknown','version':'unknown'}]})
    assert groups=={'checkout':['v1','v2']}


def test_exported_inventory_supports_simplified_pod_lists(experiment):
    path=experiment/'k8s_events/post_pods.json'
    path.parent.mkdir()
    path.write_text('[{"name":"a-pod","containers":[{"image":"demo:v1"}]}]')
    result=deployment_inventory([experiment])
    assert result['image_strings']==['demo:v1']
    assert result['frontend_inbound_rps_median'] is None


def test_control_signature_excludes_service_names_and_injection_labels(experiment):
    import json
    from dyrca_rca.experiments.audit_revised_coverage import topology_signature,declared_decoys
    before=topology_signature(experiment)
    for phase in ('pre','during'):
        path=experiment/'topology'/(phase+'.json')
        snapshot=json.loads(path.read_text())
        for pod in snapshot['pods']:pod['service']='renamed-'+pod['service']
        for service in snapshot['services']:service['name']='renamed-'+service['name']
        snapshot['root_cause_service']='CHANGED_LABEL'
        path.write_text(json.dumps(snapshot))
    assert topology_signature(experiment)==before
    assert declared_decoys({'topology_target_service':'x','topology_target_service_2':'y','control_mode':'scale'})==['x']
    assert declared_decoys({'topology_target_service':'x','topology_target_service_2':'y','control_mode':'multi'})==['x','y']


def test_actual_only_objects_reduce_fidelity_and_source_agreement(experiment, monkeypatch):
    assert fidelity({'a':1},{'a':1,'invented':2})['fidelity']==50
    view=observe(experiment)
    phase=next(f for f in view['evidence'] if f['kind']=='topology' and f['data']['phase']=='during')
    phase['data']['services'].append({'name':'invented','endpoint_ips':['unsupported']})
    monkeypatch.setattr('dyrca_rca.interaction.observation.load_observation',lambda *a,**kw:deepcopy(view))
    report=source_audit([experiment],{'dataset_id':'test','system':'test','max_rows':10000})
    assert report['records'][0]['actual_only_objects']==1
    assert report['natural_discrepancy_cases']==1
    assert report['fidelity']<100 and report['endpoint_precision']<100
    assert report['source_status_objects']==2
    assert report['source_status_correct']==1
    assert report['objects']==report['reference_objects']+report['actual_only_objects']
    assert report['actual_only_objects']==1
    assert report['objects_without_reference']>=1


def test_target_binding_rejects_changed_ground_truth(experiment):
    from dyrca_rca.experiments.backfill_revised_source import sha
    path=experiment/'ground_truth.json'
    checksum=sha(path)
    target,actual=read_frozen_target(path,checksum)
    assert target and actual==checksum
    path.write_text('{"root_cause_service":"tampered"}')
    with pytest.raises(ValueError,match='receipt-bound'):
        read_frozen_target(path,checksum)


def test_data_root_rejects_misbound_frozen_case(experiment):
    bind_cases_to_root([experiment],experiment.parent)
    with pytest.raises(ValueError,match='data-root'):
        bind_cases_to_root([experiment],experiment/'unrelated')


@pytest.mark.parametrize('available,error,message',[(17.9,RuntimeError,'Memory admission'),
                                                   (18,ValueError,'Completed frozen release')])
def test_completion_host_reserve_matches_eight_gib_worker_budget(tmp_path,monkeypatch,available,error,message):
    from types import SimpleNamespace
    from dyrca_rca.experiments.complete_revised_experiments import run
    monkeypatch.setattr('dyrca_rca.collection_safety.read_memory_state',lambda:{'available_gib':available})
    args=SimpleNamespace(data_root=tmp_path/'raw',out_dir=tmp_path/'out',seed=42,limit=1,
                         memory_cap_gib=8,system='boutique',experiment_root=tmp_path/'missing-release')
    with pytest.raises(error,match=message):run(args)
    assert not args.out_dir.exists()


def test_fresh_attempt_accepts_only_bound_launcher_files(tmp_path):
    import json
    from dyrca_rca.experiments.complete_revised_experiments import validate_fresh_output
    raw,out=tmp_path/'raw',tmp_path/'attempt';out.mkdir()
    module='dyrca_rca.experiments.complete_revised_experiments'
    spec={'module':module,'args':['--out-dir',str(out),'--data-root',str(raw)]}
    (out/'stage.json').write_text(json.dumps(spec));(out/'run.log').write_text('')
    validate_fresh_output(raw,out,module)
    (out/'revision_summary.json').write_text('{}')
    with pytest.raises(ValueError,match='already contains'):validate_fresh_output(raw,out,module)
    (out/'revision_summary.json').unlink()
    with pytest.raises(ValueError,match='does not match'):validate_fresh_output(raw,out,'wrong.module')
    spec['args'][-1]=str(tmp_path/'wrong-data');(out/'stage.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='mismatched'):validate_fresh_output(raw,out,module)


def test_launcher_metadata_does_not_allow_raw_output_or_symlinks(tmp_path):
    from dyrca_rca.experiments.complete_revised_experiments import validate_fresh_output
    with pytest.raises(ValueError,match='outside raw'):validate_fresh_output(tmp_path,tmp_path/'out','module')
    out=tmp_path/'out';out.mkdir();target=tmp_path/'target';target.write_text('{}')
    (out/'stage.json').symlink_to(target)
    with pytest.raises(ValueError,match='already contains'):validate_fresh_output(tmp_path/'raw',out,'module')
