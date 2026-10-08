"""Task-aware features; no targets or run identifiers enter the extractor."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from baseline.features import extract_features, _tokens

VERSION = "events-v1"
TYPES = ("handoff", "inform", "status", "state_update", "final", "system")
GROUPS = {
    "wait": ("waiting", "blocked", "on hold", "cannot proceed", "depends on", "await", "not arrived", "until", "still holding", "holding for"),
    "conflict": ("disagree", "contradict", "rejecting", "incorrect", "override", "correct value", "dispute", "must be", "instead of"),
    "repeat": ("re-check", "recheck", "iterating", "another pass", "once more", "again", "repeat", "inconclusive", "no progress", "looping", "one more attempt"),
    "drift": ("side note", "scope", "pivot", "new direction", "switching focus", "depriorit", "unrelated", "also look into", "matters more"),
    "recovery": ("recover", "reassign", "re-assign", "retry", "fallback", "takes over", "repairing", "mitigat", "escalat"),
    "ack": ("acknowledged", "got it", "roger", "confirmed", "accepted", "picked up", "starting", "taking "),
    "work": ("processing", "called", "in progress", "working through", "ran ", "intermediate", "collected"),
    "result": ("is done", "finished", "result stored", "output for", "closed out", "attached", "produced", "delivered", "ready:", "delivering", " complete, see"),
    "stop": ("halting", "terminat", "timeout", "nothing is moving", "no progress", "stopping", "turn budget"),
}
GENERIC = {"", "status", "finalize", "final_delivery", "recovery", "off_topic"}
SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def normalize(text):
    text = re.sub(r"\bart_[\w]+\b|\ba\d+\b", " ID ", str(text).lower())
    text = re.sub(r"\d+", " NUM ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(run):
    """Return run matrix row, per-turn rows, and normalized text document."""
    msgs = sorted(run["messages"], key=lambda m: m["t"])
    arts = sorted(run["artifacts"], key=lambda a: a.get("t", 0))
    state = sorted(run["shared_state"], key=lambda s: s.get("t", 0))
    n = len(msgs)
    excluded = {s.get("key", "") for s in state}
    excluded |= {tool for a in run["agents"] for tool in a.get("tools", [])}
    known = {m.get("intent") for m in msgs if m.get("intent") not in GENERIC}
    known |= {a.get("subtask") for a in arts if a.get("subtask") not in GENERIC}
    known.discard(None)
    at_t = defaultdict(list)
    for a in arts:
        at_t[a.get("t", -1)].append(a)
    by_id = {a.get("id"): a for a in arts}

    tasks, flags, assignment, result, normalized = [], [], [], [], []
    for m in msgs:
        text = m.get("text", "").lower()
        tokens = SNAKE.findall(text)
        task = m.get("intent", "") or ""
        if task in GENERIC:
            relevant = [x for x in tokens if x in known]
            if not relevant:
                relevant = [x for x in tokens if x not in excluded and not x.startswith("art_") and x not in GENERIC]
            if relevant:
                task = relevant[0]
        linked = [by_id[r] for r in m.get("refs", []) if r in by_id]
        linked += [a for a in at_t[m["t"]] if a.get("by") == m.get("from")]
        if task in GENERIC:
            subtasks = [a.get("subtask") for a in linked if a.get("subtask") not in GENERIC]
            if subtasks:
                task = subtasks[0]
        flag = {g: float(any(w in text for w in words)) for g, words in GROUPS.items()}
        is_result = m.get("type") == "handoff" and bool(m.get("refs") or linked or flag["result"])
        is_assign = m.get("type") == "handoff" and not is_result
        tasks.append(task)
        flags.append(flag)
        assignment.append(is_assign)
        result.append(is_result)
        normalized.append(normalize(text))

    task_events, task_assign, task_result, task_arts = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    pair_events, repeat_events, wait_pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, (m, task) in enumerate(zip(msgs, tasks)):
        # Unknown tasks stay separate rather than becoming one false giant task.
        key = task if task not in GENERIC else f"unknown:{i}"
        task_events[key].append(i)
        pair_events[(key, m.get("from"))].append(i)
        if assignment[i]:
            task_assign[key].append(i)
        if result[i]:
            task_result[key].append(i)
        if flags[i]["repeat"]:
            repeat_events[(key, m.get("from"))].append(i)
        if m.get("type") == "status" and flags[i]["wait"]:
            wait_pairs[(m.get("from"), m.get("to"))].append(i)
    for a in arts:
        task = a.get("subtask", "") or ""
        if task in GENERIC and a.get("t", -1) in at_t:
            t = int(a.get("t", -1))
            if 0 <= t < n and tasks[t] not in GENERIC:
                task = tasks[t]
        if task not in GENERIC:
            task_arts[task].append(a)

    states_by_key, states_at_t = defaultdict(list), defaultdict(list)
    for s in state:
        states_by_key[s.get("key")].append(s)
        states_at_t[int(s.get("t", -1))].append(s)
    norm_counts = Counter(normalized)
    sender_counts = Counter(m.get("from") for m in msgs)
    goal_tokens = _tokens(run.get("goal", ""))
    goal_overlap = [len(goal_tokens & _tokens(m.get("text", ""))) / max(1, len(goal_tokens)) for m in msgs]
    mutual_wait = defaultdict(list)
    for pair, indexes in wait_pairs.items():
        combined = sorted(indexes + wait_pairs.get(pair[::-1], []))
        for i in indexes:
            mutual_wait[i] = combined
    task_keyword_events = {}
    for task,event in task_events.items():
        task_keyword_events[task] = {g:[j for j in event if flags[j][g]] for g in GROUPS}
    rows = []
    for i, m in enumerate(msgs):
        key = tasks[i] if tasks[i] not in GENERIC else f"unknown:{i}"
        event = task_events[key]
        assigns, results, produced = task_assign[key], task_result[key], task_arts[key]
        own = pair_events[(key, m.get("from"))]
        repeated = repeat_events[(key, m.get("from"))]
        recoveries = task_keyword_events[key]["recovery"]
        target = m.get("to") if assignment[i] else m.get("from")
        target_arts = [a for a in produced if a.get("by") == target and a.get("t", -1) >= i and a.get("status") != "copy"]
        later_recoveries = [j for j in recoveries if j>i]
        first_recovery = min(later_recoveries,default=n)
        reciprocal = wait_pairs[(m.get("to"), m.get("from"))]
        current_wait = wait_pairs[(m.get("from"), m.get("to"))]
        current_states = states_at_t[i]
        f = {
            "pos": i, "pos_rel": i / max(1, n-1), "left": n-1-i, "length": n,
            "task_known": float(tasks[i] not in GENERIC),
            "intent_missing": float(not m.get("intent")), "refs_count": len(m.get("refs", [])),
            "tool_present": float(bool(m.get("tool"))), "text_length": len(m.get("text", "")),
            "self_message": float(m.get("from") == m.get("to")),
            "assignment": float(assignment[i]), "result": float(result[i]),
            "task_messages": len(event), "task_share": len(event)/max(1,n),
            "task_before": sum(j < i for j in event), "task_after": sum(j > i for j in event),
            "task_first": float(i == event[0]), "task_last": float(i == event[-1]),
            "task_span": event[-1]-event[0],
            "task_assignments": len(assigns), "task_results": len(results),
            "task_assign_before": sum(j < i for j in assigns), "task_assign_after": sum(j > i for j in assigns),
            "task_result_before": sum(j < i for j in results), "task_result_after": sum(j > i for j in results),
            "task_artifacts": len(produced), "task_final_arts": sum(a.get("status") == "final" for a in produced),
            "task_copy_arts": sum(a.get("status") == "copy" for a in produced),
            "task_partial_arts": sum(a.get("status") == "partial" for a in produced),
            "task_art_owners": len({a.get("by") for a in produced if a.get("status") != "copy"}),
            "task_assign_owners": len({msgs[j].get("to") for j in assigns}),
            "target_artifacts_after": len(target_arts),
            "target_delivery_before_recovery": sum(a.get("t", -1)<first_recovery for a in target_arts),
            "task_recovery_count": len(recoveries),
            "task_recovery_after": len(later_recoveries),
            "task_recovery_before": sum(j<i for j in recoveries),
            "next_recovery_distance": first_recovery-i,
            "initial_assignment_rescued": float(assignment[i] and bool(later_recoveries) and not any(a.get("t",-1)<first_recovery for a in target_arts)),
            "target_any_reply_after": sum(j>i and x.get("from") == target for j,x in enumerate(msgs)),
            "target_task_reply_after": sum(j>i and msgs[j].get("from") == target for j in event),
            "task_art_before": sum(a.get("t", -1) < i for a in produced),
            "task_art_after": sum(a.get("t", -1) >= i for a in produced),
            "task_first_result_distance": min((j-i for j in results if j >= i), default=n),
            "task_undelivered": float(not produced and not results and bool(assigns)),
            "task_sender_count": len(own), "task_sender_before": sum(j < i for j in own),
            "repeat_count": len(repeated), "repeat_first": float(bool(repeated) and repeated[0] == i),
            "repeat_before": sum(j < i for j in repeated), "repeat_after": sum(j > i for j in repeated),
            "repeat_span": repeated[-1]-repeated[0] if repeated else 0,
            "norm_count": norm_counts[normalized[i]],
            "sender_count": sender_counts[m.get("from")],
            "wait_pair_count": len(current_wait), "wait_reverse_count": len(reciprocal),
            "wait_first_pair": float(bool(current_wait) and current_wait[0] == i),
            "wait_reverse_nearest": min((abs(i-j) for j in reciprocal), default=n),
            "wait_combined_count": len(mutual_wait.get(i, [])),
            "wait_first_mutual": float(bool(mutual_wait.get(i)) and mutual_wait[i][0] == i),
            "wait_pair_span": (mutual_wait[i][-1]-mutual_wait[i][0]) if mutual_wait.get(i) else 0,
            "wait_pair_alternations": sum(msgs[b].get("from") != msgs[a].get("from") for a,b in zip(mutual_wait.get(i,[]),mutual_wait.get(i,[])[1:])),
            "goal_overlap": goal_overlap[i],
            "state_records": len(current_states),
            "state_override": sum(s.get("op") == "override" for s in current_states),
        }
        for typ in TYPES:
            f["type_"+typ] = float(m.get("type") == typ)
        f.update({"kw_"+g: v for g,v in flags[i].items()})
        for g in GROUPS:
            task_flags = task_keyword_events[key][g]
            f["task_kw_"+g] = len(task_flags)
            f["task_kw_first_"+g] = float(bool(task_flags) and task_flags[0] == i)
            f["local_kw_"+g] = sum(flags[j][g] for j in range(max(0,i-2), min(n,i+5)))
        hot_states = [states_by_key[s.get("key")] for s in current_states]
        f["state_key_writes"] = max((len(v) for v in hot_states), default=0)
        f["state_key_overrides"] = max((sum(s.get("op") == "override" for s in v) for v in hot_states), default=0)
        f["state_key_authors"] = max((len({s.get("agent") for s in v}) for v in hot_states), default=0)
        first_overrides = [next((s for s in v if s.get("op") == "override"), None) for v in hot_states]
        f["state_first_override"] = float(any(s is not None and s["t"] == i for s in first_overrides))
        for width in (3, 8):
            after = range(i, min(n, i+width))
            before = range(max(0,i-width), i)
            f[f"result_next{width}"] = sum(result[j] for j in after)
            f[f"repeat_next{width}"] = sum(flags[j]["repeat"] for j in after)
            f[f"wait_next{width}"] = sum(flags[j]["wait"] for j in after)
            f[f"goal_before{width}"] = np.mean([goal_overlap[j] for j in before]) if i else 0
            f[f"goal_after{width}"] = np.mean([goal_overlap[j] for j in after]) if n else 0
        rows.append(f)

    f_run = extract_features(run)
    f_run["intent_missing_share"] = sum(not m.get("intent") for m in msgs)/max(1,n)
    f_run["recovered_task_share"] = sum(t not in GENERIC for t in tasks)/max(1,n)
    f_run["identified_assignments"] = sum(assignment)
    f_run["identified_results"] = sum(result)
    for g in GROUPS:
        indexes = [i for i in range(n) if flags[i][g]]
        f_run[f"event_{g}_count"] = len(indexes)
        f_run[f"event_{g}_share"] = len(indexes)/max(1,n)
        f_run[f"event_{g}_first"] = indexes[0]/max(1,n) if indexes else -1
        f_run[f"event_{g}_last"] = indexes[-1]/max(1,n) if indexes else -1
        for b in range(4):
            f_run[f"event_{g}_bin{b}"] = sum(b*n/4 <= j < (b+1)*n/4 for j in indexes)/max(1,n/4)
    summary_keys = ("task_messages", "task_assignments", "task_art_owners", "repeat_count", "repeat_span", "wait_pair_count", "wait_reverse_count", "state_key_writes", "state_key_overrides", "state_key_authors", "task_undelivered", "task_final_arts")
    for key in summary_keys:
        values = [r[key] for r in rows]
        f_run["event_max_"+key] = max(values, default=0)
        f_run["event_mean_"+key] = float(np.mean(values)) if values else 0
    assign_rows = [rows[i] for i in range(n) if assignment[i]]
    f_run["assign_no_target_delivery"] = sum(r["target_artifacts_after"] == 0 for r in assign_rows)
    f_run["assign_no_target_reply"] = sum(r["target_task_reply_after"] == 0 for r in assign_rows)
    f_run["assign_no_target_delivery_share"] = f_run["assign_no_target_delivery"]/max(1,len(assign_rows))
    assigned_tasks = [t for t,indexes in task_assign.items() if indexes]
    f_run["assigned_task_coverage"] = sum(bool(task_arts[t]) for t in assigned_tasks)/max(1,len(assigned_tasks))
    f_run["assigned_task_count"] = len(assigned_tasks)
    f_run["unfulfilled_task_count"] = sum(not task_arts[t] for t in assigned_tasks)
    f_run["undelivered_but_acknowledged"] = sum(bool(r["target_task_reply_after"]) and not r["target_artifacts_after"] for r in assign_rows)
    f_run["rescued_initial_assignments"] = sum(r["initial_assignment_rescued"] for r in assign_rows)
    f_run["recovery_tasks"] = sum(bool(groups["recovery"]) for groups in task_keyword_events.values())
    f_run["unfulfilled_before_recovery"] = sum(bool(r["task_recovery_after"]) and not r["target_delivery_before_recovery"] for r in assign_rows)
    for threshold in (2, 3, 5, 7):
        f_run[f"repeat_groups_ge{threshold}"] = sum(len(v)>=threshold for v in repeat_events.values())
    f_run["mutual_wait_pairs"] = sum(bool(v) and bool(wait_pairs.get(pair[::-1])) for pair,v in wait_pairs.items())/2
    f_run["max_mutual_wait_count"] = max((len(v)+len(wait_pairs.get(pair[::-1],[])) for pair,v in wait_pairs.items() if v and wait_pairs.get(pair[::-1])), default=0)
    f_run["mutual_wait_agents"] = len({a for pair,v in wait_pairs.items() if v and wait_pairs.get(pair[::-1]) for a in pair})
    f_run["multi_art_tasks"] = sum(len([a for a in v if a.get("status") != "copy"]) >= 2 for v in task_arts.values())
    f_run["offtopic_artifacts"] = sum(a.get("subtask") == "off_topic" for a in arts)
    f_run["final_art_final"] = sum(a.get("subtask") == "final_delivery" and a.get("status") == "final" for a in arts)
    f_run["final_art_partial"] = sum(a.get("subtask") == "final_delivery" and a.get("status") == "partial" for a in arts)
    f_run["state_multiwriter_conflict_keys"] = sum(len({s.get("agent") for s in v})>1 and sum(s.get("op") == "override" for s in v)>1 for v in states_by_key.values())
    f_run["state_sorted_flips"] = sum(sum(v[j].get("value") != v[j-1].get("value") for j in range(1,len(v))) for v in states_by_key.values())
    doc = "GOAL " + normalize(run.get("goal", "")) + "\n" + "\n".join(str(m.get("type", ""))+" "+normalize(m.get("text", "")) for m in msgs)
    return f_run, rows, doc
