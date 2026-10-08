"""Protocol distinctions which change the interpretation of a run."""
from referee.relations import extract_relations


def sample(messages, artifacts=None):
    return {'run_id':'ignored', 'goal':'Verify budget', 'agents':[{'id':'a1','role':'coordinator','tools':[]},{'id':'a2','role':'operator','tools':[]}], 'messages':messages, 'artifacts':artifacts or [], 'shared_state':[], 'topology':{'type':'star','edges':[]}}


def msg(t, kind, task, text, sender='a1', recipient='a2', refs=None):
    return {'t':t,'type':kind,'intent':task,'text':text,'from':sender,'to':recipient,'refs':refs or []}


def test_acknowledgement_without_delivery_is_unfinished():
    _, rows = extract_relations(sample([msg(0,'handoff','check_budget','Assigning check_budget to a2.'),msg(1,'inform','check_budget','Acknowledged, starting check_budget.','a2','a1')]))
    assert rows[0]['rel_ack_without_work_delivery']
    assert rows[0]['rel_assignment_missing_output']


def test_unknown_ref_is_not_an_observed_artifact():
    _, rows = extract_relations(sample([msg(0,'handoff','check_budget','Delivering art_x for check_budget.','a2','a1',['art_x'])]))
    assert rows[0]['rel_delivery']
    assert rows[0]['rel_delivery_unobserved_refs'] == 1
    assert rows[0]['rel_delivery_real_refs'] == 0


def test_wait_can_depend_on_another_task():
    _, rows = extract_relations(sample([msg(0,'status','check_budget','Cannot proceed until a2 finishes verify_compliance.')]))
    assert rows[0]['rel_demand_is_other_task']
    assert rows[0]['rel_intent_body_mismatch']


def test_missing_subtask_is_recovered_from_artifact_reference():
    artifact={'id':'art_01','by':'a2','t':1,'subtask':'','status':'final','hash':'opaque'}
    _, rows=extract_relations(sample([msg(0,'handoff','check_budget','Assigning check_budget to a2.'),msg(1,'handoff','check_budget','Finished check_budget.','a2','a1',['art_01'])],[artifact]))
    assert rows[0]['rel_own_real_art'] == 1
    assert not rows[0]['rel_assignment_missing_output']
