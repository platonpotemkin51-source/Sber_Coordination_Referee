"""Semantic regressions for incomplete, interleaved run telemetry."""
from copy import deepcopy

from referee.events import prepare


def run(messages, artifacts=None, shared_state=None):
    return {"run_id":"unused", "goal":"Check budget and verify compliance", "agents":[{"id":"a1","role":"orchestrator","tools":[]},{"id":"a2","role":"analyst","tools":[]}],
            "messages":messages,"artifacts":artifacts or [],"shared_state":shared_state or [],"topology":{"type":"star","edges":[["a1","a2"]]}}


def message(t, kind, intent, text, sender="a1", receiver="a2", refs=None):
    return {"t":t,"type":kind,"intent":intent,"text":text,"from":sender,"to":receiver,"refs":refs or [],"tool":""}


def test_reply_to_another_task_does_not_resolve_handoff():
    example=run([message(0,"handoff","check_budget","Assigning check_budget to a2."),message(1,"inform","verify_compliance","Confirmed. verify_compliance is in progress.","a2","a1")])
    _, rows, _=prepare(example)
    assert rows[0]["target_any_reply_after"] == 1
    assert rows[0]["target_task_reply_after"] == 0
    assert rows[0]["task_undelivered"] == 1


def test_artifact_recovers_delivery_when_refs_and_intents_are_missing():
    example=run([message(0,"handoff","","Assigning check_budget to a2."),message(1,"handoff","","check_budget is done, artifact art_01 produced.","a2","a1")],
        [{"id":"art_01","by":"a2","t":1,"type":"budget_check","subtask":"check_budget","hash":"opaque","status":"final"}])
    features, rows, _=prepare(example)
    assert features["identified_assignments"] == 1
    assert features["identified_results"] == 1
    assert features["assign_no_target_delivery"] == 0
    assert rows[0]["task_undelivered"] == 0


def test_state_order_does_not_change_conflict_start():
    messages=[message(0,"state_update","check_budget","Recorded price=12."),message(1,"state_update","check_budget","Rejecting price; correct value is 13."),message(2,"state_update","check_budget","I disagree: price must be 12.")]
    state=[{"t":0,"agent":"a1","key":"price","op":"set","value":"12"},{"t":1,"agent":"a2","key":"price","op":"override","value":"13"},{"t":2,"agent":"a1","key":"price","op":"override","value":"12"}]
    _, rows, _=prepare(run(messages,shared_state=state[::-1]))
    assert rows[1]["state_first_override"] == 1
    assert rows[2]["state_first_override"] == 0
    assert rows[1]["state_key_authors"] == 2


def test_copy_does_not_count_as_new_work_owner():
    messages=[message(0,"handoff","check_budget","Assigning check_budget to a2."),message(1,"handoff","check_budget","Finished check_budget.","a2","a1",["art_01"])]
    artifacts=[{"id":"art_01","by":"a2","t":1,"subtask":"check_budget","hash":"same","type":"report","status":"final"},{"id":"art_02","by":"a1","t":1,"subtask":"check_budget","hash":"same","type":"report","status":"copy"}]
    features, rows, _=prepare(run(messages,artifacts))
    assert rows[0]["task_art_owners"] == 1
    assert features["multi_art_tasks"] == 0


def test_recovery_delivery_does_not_hide_initial_stalled_assignment():
    messages=[message(0,"handoff","check_budget","Assigning check_budget to a2."),message(1,"inform","check_budget","Acknowledged, starting check_budget.","a2","a1"),message(2,"handoff","check_budget","Escalating and retrying check_budget; we still have time."),message(3,"handoff","check_budget","Finished check_budget.","a2","a1",["art_01"])]
    artifact={"id":"art_01","by":"a2","t":3,"subtask":"check_budget","hash":"opaque","type":"report","status":"final"}
    features, rows, _=prepare(run(messages,[artifact]))
    assert rows[0]["target_artifacts_after"] == 1
    assert rows[0]["target_delivery_before_recovery"] == 0
    assert rows[0]["initial_assignment_rescued"] == 1
    assert features["rescued_initial_assignments"] == 1
