"""Additional protocol relations, derived only from observable run fields."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baseline.features import parse_run
from referee.data import CODE_HASH
from referee.events import GENERIC, GROUPS, SNAKE

# Relation features for non-empty runs are unchanged from the frozen model.
# Keep its provenance ID stable while allowing empty runs through the parser.
RELATIONS_HASH = "50710c3b5c6c"


def extract_relations(run):
    messages = sorted(run['messages'], key=lambda m: m['t'])
    n = len(messages)
    agents = {a['id']: a for a in run['agents']}
    artifacts = run['artifacts']
    by_id = {a['id']: a for a in artifacts}
    excluded = {s.get('key', '') for s in run['shared_state']}
    excluded.update(tool for a in agents.values() for tool in a.get('tools', []))
    known = {m.get('intent') for m in messages} | {a.get('subtask') for a in artifacts}
    known.difference_update(GENERIC | {None})
    tasks, mentioned, flags, assignments, deliveries = [], [], [], [], []
    for m in messages:
        text = m.get('text', '').lower()
        tokens = [x for x in SNAKE.findall(text) if x not in excluded and not x.startswith('art_') and x not in GENERIC]
        tokens.sort(key=lambda x: x not in known)
        body_task = tokens[0] if tokens else ''
        task = m.get('intent', '')
        if task in GENERIC:
            task = body_task
        linked = [by_id[r] for r in m.get('refs', []) if r in by_id]
        if not task:
            task = next((a.get('subtask') for a in linked if a.get('subtask') not in GENERIC), '')
        flag = {g: any(w in text for w in words) for g, words in GROUPS.items()}
        delivery = m.get('type') == 'handoff' and bool(m.get('refs') or linked or flag['result'] or 'art_' in text)
        tasks.append(task if task not in GENERIC else '')
        mentioned.append(body_task)
        flags.append(flag)
        deliveries.append(delivery)
        assignments.append(m.get('type') == 'handoff' and not delivery)

    # Missing intent and truncated text can be resolved by an unambiguous
    # earlier assignment to this sender. Multiple pending tasks stay unknown.
    pending = defaultdict(set)
    for i, m in enumerate(messages):
        if assignments[i] and tasks[i]:
            pending[m.get('to')].add(tasks[i])
        if not tasks[i] and m.get('intent', '') == '' and len(pending[m.get('from')]) == 1:
            tasks[i] = next(iter(pending[m.get('from')]))
        if deliveries[i] and tasks[i]:
            pending[m.get('from')].discard(tasks[i])

    events, assigns, results, produced = (defaultdict(list) for _ in range(4))
    for i, task in enumerate(tasks):
        if not task:
            continue
        events[task].append(i)
        if assignments[i]:
            assigns[task].append(i)
        if deliveries[i]:
            results[task].append(i)
    for artifact in artifacts:
        task = artifact.get('subtask', '')
        if task in GENERIC:
            referenced = [i for i, m in enumerate(messages) if artifact['id'] in m.get('refs', []) or artifact['id'] in m.get('text', '')]
            matched = [i for i in referenced if messages[i].get('from') == artifact.get('by') and tasks[i]]
            if not matched:
                matched = [i for i, m in enumerate(messages) if m['t'] == artifact.get('t') and m.get('from') == artifact.get('by') and tasks[i]]
            task = tasks[matched[0]] if matched else ''
        if task and task not in GENERIC:
            produced[task].append(artifact)
    pair_counts = Counter((m.get('from'), m.get('to')) for m in messages)
    state_by_key = defaultdict(list)
    for state in sorted(run['shared_state'], key=lambda s: s.get('t', -1)):
        state_by_key[state.get('key')].append(state)
    rows = []
    for i, m in enumerate(messages):
        task = tasks[i]
        indexes, aa, rr, art = events[task], assigns[task], results[task], produced[task]
        sender, recipient = m.get('from'), m.get('to')
        owner = recipient if assignments[i] else sender
        own = [j for j in indexes if messages[j].get('from') == owner]
        future = [j for j in own if j > i]
        own_deliveries = [j for j in rr if messages[j].get('from') == owner]
        future_deliveries = [j for j in own_deliveries if j >= i]
        real_art = [a for a in art if a.get('status') != 'copy']
        own_art = [a for a in real_art if a.get('by') == owner]
        demand = mentioned[i] or task
        demand_art = [a for a in produced[demand] if a.get('status') != 'copy']
        current = [s for s in run['shared_state'] if s.get('t') == m['t']]
        keys = [state_by_key[s.get('key')] for s in current]
        f = {
            'rel_task_known': bool(task),
            'rel_intent_body_mismatch': bool(task and mentioned[i] and task != mentioned[i]),
            'rel_demand_is_other_task': bool(task and demand and task != demand),
            'rel_sender_coordinator': agents.get(sender, {}).get('role') in ('coordinator', 'orchestrator'),
            'rel_recipient_coordinator': agents.get(recipient, {}).get('role') in ('coordinator', 'orchestrator'),
            'rel_sender_tools': len(agents.get(sender, {}).get('tools', [])),
            'rel_tool_declared': bool(m.get('tool')) and m.get('tool') in agents.get(sender, {}).get('tools', []),
            'rel_pair_messages': pair_counts[sender, recipient],
            'rel_pair_reverse': pair_counts[recipient, sender],
            'rel_assignment': assignments[i], 'rel_delivery': deliveries[i],
            'rel_task_assignments': len(aa),
            'rel_task_assigned': bool(aa),
            'rel_sender_assigned_task': any(messages[j].get('to') == sender for j in aa),
            'rel_recipient_assigned_task': any(messages[j].get('to') == recipient for j in aa),
            'rel_assignment_ordinal': sum(j <= i for j in aa),
            'rel_assignment_other_owner_before': sum(j < i and messages[j].get('to') != owner for j in aa),
            'rel_own_ack_after': sum(flags[j]['ack'] for j in future),
            'rel_own_work_after': sum(flags[j]['work'] for j in future),
            'rel_own_wait_after': sum(flags[j]['wait'] for j in future),
            'rel_own_repeat_after': sum(flags[j]['repeat'] for j in future),
            'rel_own_delivery_after': len(future_deliveries),
            'rel_own_art_after': sum(a.get('t', -1) >= i for a in own_art),
            'rel_own_real_art': len(own_art),
            'rel_own_reply_after': len(future),
            'rel_own_reply_first_distance': min((j-i for j in future), default=n),
            'rel_own_delivery_distance': min((j-i for j in future_deliveries), default=n),
            'rel_delivery_real_refs': sum(r in by_id for r in m.get('refs', [])),
            'rel_delivery_unobserved_refs': sum(r not in by_id for r in m.get('refs', [])),
            'rel_real_task_artifacts': len(real_art),
            'rel_real_task_art_owners': len({a.get('by') for a in real_art}),
            'rel_real_duplicate_hashes': len(real_art)-len({a.get('hash') for a in real_art}),
            'rel_task_delivery_before': sum(j < i for j in rr),
            'rel_task_art_before': sum(a.get('t', -1) < i for a in real_art),
            'rel_task_first_distance': i-indexes[0] if indexes else n,
            'rel_task_final_distance': indexes[-1]-i if indexes else n,
            'rel_demand_delivered': bool(demand_art),
            'rel_demand_art_after': sum(a.get('t', -1) > i for a in demand_art),
            'rel_demand_delivery_target_after': sum(a.get('t', -1) > i and a.get('by') == recipient for a in demand_art),
            'rel_state_value_alternations': max((sum(v[j].get('value') == v[j-2].get('value') and v[j].get('value') != v[j-1].get('value') for j in range(2,len(v))) for v in keys), default=0),
            'rel_state_author_alternations': max((sum(v[j].get('agent') != v[j-1].get('agent') for j in range(1,len(v))) for v in keys), default=0),
            'rel_state_write_ordinal': max((sum(s.get('t', -1) <= i for s in v) for v in keys), default=0),
        }
        f['rel_ack_without_work_delivery'] = assignments[i] and f['rel_own_ack_after'] > 0 and f['rel_own_work_after'] == 0 and not future_deliveries and not own_art
        f['rel_assignment_missing_output'] = assignments[i] and not future_deliveries and not own_art
        for group in ('wait','conflict','repeat','drift'):
            signals = [j for j in indexes if flags[j][group]]
            f['rel_task_'+group+'_count'] = len(signals)
            f['rel_task_'+group+'_before'] = sum(j < i for j in signals)
            f['rel_task_'+group+'_first'] = bool(signals) and signals[0] == i
            f['rel_task_'+group+'_unassigned'] = len(signals) if not aa else 0
        rows.append(f)
    summary = {}
    for key in rows[0] if rows else []:
        values = np.array([r[key] for r in rows], dtype=np.float32)
        summary[key+'_max'] = float(values.max(initial=0))
        summary[key+'_mean'] = float(values.mean()) if n else 0
        if 'assignment' in key or 'without_work' in key:
            summary[key+'_sum'] = float(values.sum())
    return summary, rows


def relation_features(data, source, cache='outputs/cache'):
    target = Path(cache)/f"relations-{data['source_sha256'][:16]}-{CODE_HASH}-{RELATIONS_HASH}.joblib"
    if target.exists():
        values = joblib.load(target)
    else:
        run_rows, turn_rows, columns = [], [], None
        frame = pd.read_csv(source)
        for i, record in enumerate(frame.to_dict('records')):
            run, turns = extract_relations(parse_run(record))
            if columns is None and turns:
                columns = sorted(turns[0])
            run_rows.append(run)
            turn_rows.append(np.array([[row[c] for c in columns] for row in turns], dtype=np.float32) if turns else None)
            if i % 1000 == 0:
                print(f'relations {i}/{len(frame)}', flush=True)
        if columns is None:
            raise ValueError('Cannot infer relation feature columns: the input contains no messages in any run')
        turn_rows = [rows if rows is not None else np.zeros((0, len(columns)), dtype=np.float32) for rows in turn_rows]
        values = {'run': pd.DataFrame(run_rows).fillna(0).astype(np.float32), 'turn': np.concatenate(turn_rows), 'turn_columns': columns}
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(values, target, compress=3)
    if len(values['turn']) != len(data['turn_X']):
        raise ValueError('Relation events do not align with the original turns')
    out = dict(data)
    out['X'] = pd.concat([data['X'], values['run']], axis=1)
    out['turn_X'] = np.hstack([data['turn_X'], values['turn']])
    out['turn_columns'] = data['turn_columns']+values['turn_columns']
    return out
