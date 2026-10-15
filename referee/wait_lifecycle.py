"""Observable lifetime of wait dependencies; experimental, not in submission.

Edges are evidence of waiting, not proof of a primary deadlock. Deliveries and
resumed work close compatible edges. Only observed timestamps and text are used.
"""
from collections import defaultdict
import numpy as np

from referee.events import GENERIC, GROUPS, SNAKE


def extract_wait_lifecycle(run, return_steps=False):
    messages = sorted(run["messages"], key=lambda m: m["t"])
    artifacts = sorted(run["artifacts"], key=lambda a: a.get("t", 0))
    by_id = {a.get("id"): a for a in artifacts}
    known = {m.get("intent") for m in messages} | {a.get("subtask") for a in artifacts}
    known -= GENERIC | {None}
    edges, closed = {}, []
    snapshots = []
    artifact_pos = 0
    cycle_age = longest_cycle = cycle_starts = cycle_breaks = 0
    resolutions = defaultdict(int)

    def close_matching(predicate, index, reason):
        for key in list(edges):
            if predicate(key, edges[key]):
                closed.append(index - edges.pop(key)["start"])
                resolutions[reason] += 1

    for index, message in enumerate(messages):
        # Same-turn artifacts are visible when processing the associated event.
        while artifact_pos < len(artifacts) and artifacts[artifact_pos].get("t", 0) <= message["t"]:
            artifact = artifacts[artifact_pos]
            task, producer = artifact.get("subtask"), artifact.get("by")
            if task in known and artifact.get("status") != "copy":
                close_matching(lambda key, edge: key[1] == producer and key[2] == task,
                               index, "artifact")
            artifact_pos += 1
        text = message.get("text", "").lower()
        flags = {group: any(word in text for word in words) for group, words in GROUPS.items()}
        sender, recipient = message.get("from"), message.get("to")
        own_task = message.get("intent") or ""
        mentions = [token for token in SNAKE.findall(text) if token in known]
        demand = mentions[0] if mentions else (own_task if own_task in known else "")
        linked_tasks = {by_id[ref].get("subtask") for ref in message.get("refs", []) if ref in by_id}
        delivered = (message.get("type") == "handoff" and
                     bool(message.get("refs") or flags["result"]))
        if delivered:
            completed = linked_tasks | ({own_task} if own_task in known else set())
            close_matching(lambda key, edge: key[1] == sender and key[2] in completed,
                           index, "delivery")
        if (flags["work"] or delivered) and own_task in known and not flags["wait"]:
            close_matching(lambda key, edge: key[0] == sender and edge["own_task"] == own_task,
                           index, "resumed")
        if flags["wait"] and sender and recipient and sender != recipient:
            key = (sender, recipient, demand)
            if key not in edges:
                edges[key] = {"start": index, "own_task": own_task, "count": 0}
            edges[key]["count"] += 1
        graph = defaultdict(set)
        for source, target, _ in edges:
            graph[source].add(target)
        nodes = set(graph) | {node for neighbours in graph.values() for node in neighbours}
        reach = {}
        for node in nodes:
            seen, stack = set(), list(graph[node])
            while stack:
                nxt = stack.pop()
                if nxt not in seen:
                    seen.add(nxt)
                    stack.extend(graph[nxt] - seen)
            reach[node] = seen
        cyclic = {node for node in nodes if node in reach[node]}
        impacted = {node for node in nodes if reach[node] & cyclic}
        if cyclic:
            cycle_starts += cycle_age == 0
            cycle_age += 1
            longest_cycle = max(longest_cycle, cycle_age)
        else:
            cycle_breaks += cycle_age > 0
            cycle_age = 0
        ages = [index - edge["start"] for edge in edges.values()]
        snapshots.append({
            "active_edges": len(edges), "cyclic_agents": len(cyclic),
            "impacted_agents": len(impacted), "cycle_age": cycle_age,
            "oldest_edge": max(ages, default=0),
            "unknown_task_edges": sum(not key[2] for key in edges),
            "repeated_wait_edges": sum(edge["count"] > 1 for edge in edges.values()),
        })
    result = {}
    keys = ("active_edges", "cyclic_agents", "impacted_agents", "cycle_age", "oldest_edge",
            "unknown_task_edges", "repeated_wait_edges")
    for key in keys:
        values = np.asarray([row[key] for row in snapshots], dtype=float)
        result[f"life_{key}_max"] = float(values.max(initial=0))
        result[f"life_{key}_mean"] = float(values.mean()) if len(values) else 0.
        result[f"life_{key}_end"] = float(values[-1]) if len(values) else 0.
    result.update({"life_cycle_starts": cycle_starts, "life_cycle_breaks": cycle_breaks,
                   "life_longest_cycle_fraction": longest_cycle / max(1, len(messages)),
                   "life_closed_wait_count": len(closed),
                   "life_closed_wait_mean": float(np.mean(closed)) if closed else 0.,
                   "life_closed_wait_max": max(closed, default=0),
                   **{f"life_resolved_by_{kind}": resolutions[kind]
                      for kind in ("artifact", "delivery", "resumed")}})
    return (result, snapshots) if return_steps else result
