"""Feature extraction for the Coordination Referee baseline.

One run (a row of ``train.csv`` / ``test.csv``) -> one flat dict of numeric
features.  Pure Python + stdlib, no graph library required: everything runs
comfortably inside 4 vCPU / 16 GB.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

PROTOCOL_TYPES = ["handoff", "inform", "status", "state_update", "final", "system"]
TOPOLOGY_TYPES = ["star", "pipeline", "mesh", "hierarchical", "blackboard"]

KEYWORDS = {
    "wait": ["wait", "blocked", "hold", "depend", "not arrived", "cannot proceed", "on hold"],
    "conflict": ["disagree", "contradict", "overrid", "incorrect", "reject", "conflict", "wrong"],
    "repeat": ["again", "re-check", "recheck", "iterat", "repeat", "another pass",
               "looping", "one more", "still not"],
    "drift": ["side note", "expanding scope", "pivot", "new direction", "switching focus",
              "deprioritis", "deprioritiz", "also look into", "matters more"],
    "recovery": ["recover", "re-assign", "reassign", "retry", "fallback", "escalat",
                 "mitigat", "repairing", "takes over"],
    "stop": ["terminat", "stops the run", "halting", "stopping the run", "budget of turns",
             "no progress", "turn budget"],
    "done": ["final", "complete", "deliver", "ready", "wrapping up", "consolidat",
             "aggregated"],
}

_WORD = re.compile(r"[a-z]+")
_STOP = {"the", "a", "an", "for", "and", "of", "to", "in", "on", "with", "by", "is",
         "be", "all", "from", "before", "within", "end", "into", "as", "it", "we"}


def _norm(text: str) -> str:
    """Normalised message text: lowercase, digits and ids removed."""
    t = text.lower()
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"art_\S+|a\d+", "@", t)
    return re.sub(r"[^a-z@#\s]", " ", t).strip()


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts if c > 0)


def _safe_div(a, b) -> float:
    return float(a) / b if b else 0.0


def parse_run(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decode the JSON columns of one CSV row."""
    out = {"run_id": row["run_id"], "goal": row.get("goal", "")}
    for f in ("agents", "shared_state", "messages", "artifacts", "topology"):
        v = row.get(f)
        out[f] = json.loads(v) if isinstance(v, str) else (v if v is not None else [])
    return out


# ---------------------------------------------------------------------------
# graph helpers (tiny hand-rolled implementations, no networkx dependency)
# ---------------------------------------------------------------------------
def _largest_scc(nodes, adj) -> int:
    """Tarjan SCC, iterative; returns the size of the largest component."""
    index, low, on_stack, stack, comp = {}, {}, set(), [], []
    counter = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(adj.get(nxt, ()))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                size = 0
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    size += 1
                    if w == node:
                        break
                comp.append(size)
    return max(comp) if comp else 0


def _graph_features(messages) -> Dict[str, float]:
    adj = defaultdict(set)
    pair = Counter()
    nodes = set()
    self_loops = 0
    for m in messages:
        s, r = m.get("from"), m.get("to")
        if not s or not r:
            continue
        nodes.add(s)
        nodes.add(r)
        pair[(s, r)] += 1
        if s == r:
            self_loops += 1
        else:
            adj[s].add(r)
    n = len(nodes)
    edges = sum(len(v) for v in adj.values())
    recips = sum(1 for (s, r) in {(a, b) for a in adj for b in adj[a]}
                 if s in adj.get(r, ()))
    two_cycles = recips // 2
    triangles = 0
    for a in adj:
        for b in adj[a]:
            for c in adj.get(b, ()):
                if a in adj.get(c, ()) and a != b and b != c and a != c:
                    triangles += 1
    out_deg = [len(adj.get(x, ())) for x in nodes] or [0]
    in_deg = Counter()
    for a in adj:
        for b in adj[a]:
            in_deg[b] += 1
    max_pair = max(pair.values()) if pair else 0
    top_pair_share = _safe_div(max_pair, sum(pair.values()))
    pingpong = max([min(pair[(a, b)], pair[(b, a)]) for (a, b) in pair] or [0])
    return {
        "g_nodes": n, "g_edges": edges,
        "g_density": _safe_div(edges, n * (n - 1)) if n > 1 else 0.0,
        "g_self_loops": self_loops,
        "g_reciprocity": _safe_div(recips, edges),
        "g_two_cycles": two_cycles, "g_triangles": triangles / 3.0,
        "g_max_out_deg": max(out_deg), "g_mean_out_deg": sum(out_deg) / len(out_deg),
        "g_max_in_deg": max(in_deg.values()) if in_deg else 0,
        "g_largest_scc": _largest_scc(nodes, adj),
        "g_scc_share": _safe_div(_largest_scc(nodes, adj), n),
        "g_n_pairs": len(pair), "g_max_pair": max_pair,
        "g_top_pair_share": top_pair_share, "g_pingpong": pingpong,
        "g_pingpong_share": _safe_div(pingpong * 2, len(messages)),
    }


# ---------------------------------------------------------------------------
def extract_features(run: Dict[str, Any]) -> Dict[str, float]:
    msgs: List[Dict] = run["messages"]
    agents = run["agents"]
    arts = run["artifacts"]
    state = run["shared_state"]
    topo = run["topology"] if isinstance(run["topology"], dict) else {}
    n = len(msgs)
    f: Dict[str, float] = {}

    # ---- volume ----
    f["n_messages"] = n
    f["n_agents"] = len(agents)
    f["n_artifacts"] = len(arts)
    f["n_state_writes"] = len(state)
    f["msgs_per_agent"] = _safe_div(n, len(agents))
    f["arts_per_msg"] = _safe_div(len(arts), n)
    f["state_per_msg"] = _safe_div(len(state), n)
    f["n_tools_declared"] = sum(len(a.get("tools", [])) for a in agents)

    # ---- protocol types ----
    tcount = Counter(m.get("type", "") for m in msgs)
    for t in PROTOCOL_TYPES:
        f["type_%s" % t] = tcount.get(t, 0)
        f["share_%s" % t] = _safe_div(tcount.get(t, 0), n)
    f["has_final_msg"] = 1.0 if tcount.get("final") else 0.0
    f["has_system_msg"] = 1.0 if tcount.get("system") else 0.0

    # ---- senders ----
    senders = Counter(m.get("from") for m in msgs)
    f["n_senders"] = len(senders)
    f["sender_entropy"] = _entropy(list(senders.values()))
    f["sender_max_share"] = _safe_div(max(senders.values()) if senders else 0, n)
    active = {m.get("from") for m in msgs} | {m.get("to") for m in msgs}
    f["silent_agent_ratio"] = _safe_div(
        sum(1 for a in agents if a["id"] not in active), len(agents))

    # ---- repetition ----
    texts = [m.get("text", "") for m in msgs]
    norms = [_norm(t) for t in texts]
    ex = Counter(texts)
    nm = Counter(norms)
    f["dup_text_max"] = max(ex.values()) if ex else 0
    f["dup_text_ratio"] = _safe_div(n - len(ex), n)
    f["norm_dup_max"] = max(nm.values()) if nm else 0
    f["norm_dup_ratio"] = _safe_div(n - len(nm), n)
    f["norm_unique_ratio"] = _safe_div(len(nm), n)
    streak = best = 0
    prev = None
    for x in norms:
        streak = streak + 1 if x == prev else 1
        best = max(best, streak)
        prev = x
    f["max_consecutive_repeat"] = best
    pair_streak = best_pair = 0
    prev_pair = None
    for m, x in zip(msgs, norms):
        key = (m.get("from"), x)
        pair_streak = pair_streak + 1 if key == prev_pair else 1
        best_pair = max(best_pair, pair_streak)
        prev_pair = key
    f["max_sender_repeat_streak"] = best_pair

    intents = Counter(m.get("intent", "") for m in msgs)
    f["n_intents"] = len(intents)
    f["intent_entropy"] = _entropy(list(intents.values()))
    f["intent_max_count"] = max(intents.values()) if intents else 0
    f["intent_max_share"] = _safe_div(f["intent_max_count"], n)

    # ---- lexical ----
    blob = " ".join(texts).lower()
    for grp, words in KEYWORDS.items():
        c = sum(blob.count(w) for w in words)
        f["kw_%s" % grp] = c
        f["kw_%s_rate" % grp] = _safe_div(c, n)

    # ---- goal alignment ----
    goal_tok = _tokens(run.get("goal", ""))
    overlaps = [_safe_div(len(goal_tok & _tokens(t)), len(goal_tok) or 1) for t in texts]
    if overlaps:
        third = max(1, n // 3)
        f["goal_ov_mean"] = sum(overlaps) / n
        f["goal_ov_head"] = sum(overlaps[:third]) / third
        f["goal_ov_tail"] = sum(overlaps[-third:]) / third
        f["goal_ov_delta"] = f["goal_ov_tail"] - f["goal_ov_head"]
        f["goal_ov_zero_ratio"] = _safe_div(sum(1 for o in overlaps if o == 0), n)
        f["goal_ov_max"] = max(overlaps)
    else:
        for k in ("goal_ov_mean", "goal_ov_head", "goal_ov_tail", "goal_ov_delta",
                  "goal_ov_zero_ratio", "goal_ov_max"):
            f[k] = 0.0
    head_intents = {m.get("intent") for m in msgs[:max(1, n // 3)]}
    tail_intents = {m.get("intent") for m in msgs[-max(1, n // 3):]}
    f["intent_head_tail_jaccard"] = _safe_div(
        len(head_intents & tail_intents), len(head_intents | tail_intents))
    f["new_intents_in_tail"] = len(tail_intents - head_intents)

    # ---- handoff / delivery accounting ----
    assigns, results = [], []
    for m in msgs:
        if m.get("type") == "handoff":
            (results if m.get("refs") else assigns).append(m)
    delivered_by = defaultdict(set)          # agent -> delivered intents
    for m in results:
        delivered_by[m.get("from")].add(m.get("intent"))
    for a in arts:
        delivered_by[a.get("by")].add(a.get("subtask"))

    unanswered = undelivered = 0
    delays = []
    for m in assigns:
        t, to = m.get("t", 0), m.get("to")
        reply_t = next((x.get("t") for x in msgs
                        if x.get("t", 0) > t and x.get("from") == to), None)
        if reply_t is None:
            unanswered += 1
        else:
            delays.append(reply_t - t)
        if m.get("intent") not in delivered_by.get(to, set()):
            undelivered += 1
    f["n_assign"] = len(assigns)
    f["n_result"] = len(results)
    f["assign_result_gap"] = len(assigns) - len(results)
    f["unanswered_assign"] = unanswered
    f["unanswered_ratio"] = _safe_div(unanswered, len(assigns))
    f["undelivered_assign"] = undelivered
    f["undelivered_ratio"] = _safe_div(undelivered, len(assigns))
    f["reply_delay_mean"] = _safe_div(sum(delays), len(delays))
    f["reply_delay_max"] = max(delays) if delays else 0

    assigned_intents = Counter(m.get("intent") for m in assigns)
    f["reassigned_intents"] = sum(1 for v in assigned_intents.values() if v > 1)
    f["reassigned_ratio"] = _safe_div(f["reassigned_intents"], max(1, len(assigned_intents)))
    f["max_assign_per_intent"] = max(assigned_intents.values()) if assigned_intents else 0
    multi_owner = sum(1 for i, owners in
                      {i: {m.get("to") for m in assigns if m.get("intent") == i}
                       for i in assigned_intents}.items() if len(owners) > 1)
    f["intents_with_multiple_owners"] = multi_owner

    # ---- artifacts ----
    hashes = Counter(a.get("hash") for a in arts)
    f["n_unique_hashes"] = len(hashes)
    f["dup_hash_pairs"] = sum(v - 1 for v in hashes.values() if v > 1)
    f["dup_hash_max"] = max(hashes.values()) if hashes else 0
    f["dup_hash_ratio"] = _safe_div(f["dup_hash_pairs"], len(arts))
    sub_counter = Counter(a.get("subtask") for a in arts)
    f["max_arts_per_subtask"] = max(sub_counter.values()) if sub_counter else 0
    f["subtasks_with_2plus_arts"] = sum(1 for v in sub_counter.values() if v > 1)
    st = Counter(a.get("status") for a in arts)
    f["art_partial"] = st.get("partial", 0)
    f["art_copy"] = st.get("copy", 0)
    f["has_final_artifact"] = 1.0 if any(
        a.get("subtask") == "final_delivery" for a in arts) else 0.0
    f["coverage"] = _safe_div(len({a.get("subtask") for a in arts} & set(assigned_intents)),
                              max(1, len(assigned_intents)))
    f["art_types"] = len({a.get("type") for a in arts})

    # ---- shared state ----
    per_key = defaultdict(list)
    overrides = 0
    for s in state:
        per_key[s.get("key")].append(s.get("value"))
        overrides += 1 if s.get("op") == "override" else 0
    flips = alt = 0
    for vals in per_key.values():
        for i in range(1, len(vals)):
            if vals[i] != vals[i - 1]:
                flips += 1
            if i >= 2 and vals[i] == vals[i - 2] and vals[i] != vals[i - 1]:
                alt += 1
    f["state_keys"] = len(per_key)
    f["state_overrides"] = overrides
    f["state_override_ratio"] = _safe_div(overrides, len(state))
    f["state_max_writes_per_key"] = max((len(v) for v in per_key.values()), default=0)
    f["state_max_distinct_per_key"] = max((len(set(v)) for v in per_key.values()), default=0)
    f["state_flips"] = flips
    f["state_alternating_flips"] = alt
    f["state_keys_3plus"] = sum(1 for v in per_key.values() if len(v) >= 3)
    status = per_key.get("run_status", [])
    f["status_completed"] = 1.0 if "completed" in status else 0.0
    f["status_failed"] = 1.0 if any(v in ("timeout", "aborted", "incomplete")
                                    for v in status) else 0.0
    f["status_missing"] = 0.0 if status else 1.0

    # ---- topology ----
    ttype = topo.get("type", "")
    for t in TOPOLOGY_TYPES:
        f["topo_%s" % t] = 1.0 if ttype == t else 0.0
    edges = topo.get("edges", [])
    f["topo_edges"] = len(edges)
    f["topo_edge_per_agent"] = _safe_div(len(edges), len(agents))

    # ---- tail behaviour ----
    tail = msgs[-max(1, n // 5):] if n else []
    tc = Counter(m.get("type") for m in tail)
    f["tail_status_share"] = _safe_div(tc.get("status", 0), len(tail))
    f["tail_inform_share"] = _safe_div(tc.get("inform", 0), len(tail))
    f["tail_handoff_share"] = _safe_div(tc.get("handoff", 0), len(tail))
    f["tail_has_final"] = 1.0 if tc.get("final") else 0.0

    f.update(_graph_features(msgs))
    return f
