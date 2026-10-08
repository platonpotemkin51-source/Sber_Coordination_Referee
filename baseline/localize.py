"""Rule-based localisation of the fault turn (bonus metric ``fault_turn_hit@2``).

Given a run and the *predicted* fault class, return the index of the message
where that fault most plausibly starts, or ``-1`` for ``clean``.
The baseline deliberately keeps this heuristic: beating it with a learned
per-turn ranker is one of the obvious ways for participants to gain points.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

try:
    from .features import _norm, _tokens
except ImportError:  # Direct execution by the original starter scripts.
    from features import _norm, _tokens


def _assigns_and_results(msgs):
    assigns = [m for m in msgs if m.get("type") == "handoff" and not m.get("refs")]
    results = [m for m in msgs if m.get("type") == "handoff" and m.get("refs")]
    return assigns, results


def localize(run: Dict[str, Any], label: str) -> int:
    msgs: List[Dict] = run["messages"]
    if not msgs or label == "clean":
        return -1
    arts = run["artifacts"]
    state = run["shared_state"]
    assigns, results = _assigns_and_results(msgs)

    if label == "dropped_handoff":
        delivered = defaultdict(set)
        for m in results:
            delivered[m.get("from")].add(m.get("intent"))
        for a in arts:
            delivered[a.get("by")].add(a.get("subtask"))
        for m in assigns:
            if m.get("intent") not in delivered.get(m.get("to"), set()):
                return int(m.get("t", -1))
        for m in assigns:
            t, to = m.get("t", 0), m.get("to")
            if not any(x.get("from") == to and x.get("t", 0) > t for x in msgs):
                return int(m.get("t", -1))
        return int(assigns[0]["t"]) if assigns else -1

    if label == "duplicated_work":
        seen = {}
        for m in assigns:
            key = m.get("intent")
            if key in seen:
                return int(m.get("t", -1))
            seen[key] = m.get("t")
        hashes = Counter(a.get("hash") for a in arts)
        dup = [a for a in arts if hashes[a.get("hash")] > 1]
        if dup:
            return int(sorted(dup, key=lambda a: a.get("t", 0))[-1].get("t", -1))
        return int(msgs[len(msgs) // 2]["t"])

    if label == "deadlock":
        best_t, best_len, cur_len, cur_start = -1, 0, 0, -1
        for m in msgs:
            if m.get("type") == "status":
                if cur_len == 0:
                    cur_start = int(m.get("t", 0))
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_t = cur_len, cur_start
            else:
                cur_len = 0
        return best_t if best_t >= 0 else -1

    if label == "conflict":
        per_key = defaultdict(list)
        for s in state:
            per_key[s.get("key")].append(s)
        hot = max(per_key.items(), key=lambda kv: len(kv[1]), default=(None, []))
        if hot[0] is not None and len(hot[1]) >= 2:
            for s in hot[1]:
                if s.get("op") == "override":
                    return int(s.get("t", -1))
        for m in msgs:
            if m.get("type") == "state_update" and "gree" in m.get("text", "").lower():
                return int(m.get("t", -1))
        return -1

    if label == "goal_drift":
        goal_tok = _tokens(run.get("goal", ""))
        zeros = [int(m.get("t", 0)) for m in msgs
                 if not (goal_tok & _tokens(m.get("text", "")))]
        for i in range(len(zeros) - 2):
            if zeros[i + 2] - zeros[i] <= 6:
                return zeros[i]
        return zeros[0] if zeros else -1

    if label == "runaway_loop":
        norms = [_norm(m.get("text", "")) for m in msgs]
        counts = Counter(norms)
        hot = {t for t, c in counts.items() if c >= 3}
        best_t, best_len, cur_len, cur_start = -1, 0, 0, -1
        for i, t in enumerate(norms):
            if t in hot:
                if cur_len == 0:
                    cur_start = int(msgs[i].get("t", i))
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_t = cur_len, cur_start
            else:
                cur_len = max(0, cur_len - 1)
                if cur_len == 0:
                    cur_start = -1
        if best_t >= 0:
            return best_t
        intents = Counter(m.get("intent") for m in msgs)
        top = intents.most_common(1)[0][0]
        return int(next(m["t"] for m in msgs if m.get("intent") == top))

    return -1
