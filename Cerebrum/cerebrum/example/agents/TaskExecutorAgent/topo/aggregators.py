from typing import List, Dict, Any
import json
import networkx as nx

def _nameset(G: nx.Graph) -> set[str]:
    """返回图中所有节点名字集合"""
    return {str(G.nodes[i]["name"]) for i in G.nodes()}

def _name(G: nx.Graph, i) -> str:
    """节点 id -> name"""
    return str(G.nodes[i]["name"])

# 统一把图转成无向图（防止DAG误判相邻）
def _undirected(G: nx.Graph) -> nx.Graph:
    return G.to_undirected() if G.is_directed() else G

def agg_coloring(agent_jsons: List[Dict[str, Any]], G: nx.Graph) -> Dict[str, Any]:
    """
    agent_jsons: [{"node":"Hannah","channel":1}, ...]
    return: {"channels": {name:int}, "valid": bool}
    """
    channels: Dict[str, int] = {}
    valid_names = _nameset(G)
    for rec in agent_jsons:
        try:
            node_name = str(rec["node"])
            ch = int(rec["channel"])
        except Exception:
            continue
        if node_name in valid_names:
            channels[node_name] = ch

    UG = _undirected(G)
    ok = True
    for u, v in UG.edges():
        nu, nv = _name(G, u), _name(G, v)
        cu, cv = channels.get(nu), channels.get(nv)
        if cu is None or cv is None or cu == cv:
            ok = False
            break
    return {"channels": channels, "valid": ok}

def agg_consensus(agent_jsons: List[Dict[str, Any]], G: nx.Graph) -> Dict[str, Any]:
    """
    agent_jsons: [{"node":"James","value":"A"}, ...]
    return: {"value": <any>, "agreement": bool}
    """
    values = []
    names = _nameset(G)
    for rec in agent_jsons:
        n = str(rec.get("node", ""))
        if n not in names:
            continue
        if "value" in rec:
            values.append(rec["value"])
        elif "decided" in rec:
            values.append(rec["decided"])
    if not values:
        return {"value": None, "agreement": False}

    try:
        canon = {json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values}
    except Exception:
        canon = set(map(str, values))

    agreement = len(canon) == 1
    value = values[0] if agreement else None
    return {"value": value, "agreement": agreement}

def agg_leader_election(agent_jsons: List[Dict[str, Any]], G: nx.Graph) -> Dict[str, Any]:
    """
    agent_jsons: [{"node":"A","leader":"B"}, ...]
    return: {"leader": <name>, "votes": {<name>:int}}
    """
    votes: Dict[str, int] = {}
    names = _nameset(G)
    for rec in agent_jsons:
        voter = str(rec.get("node", ""))
        if voter not in names:
            continue
        cand = str(rec.get("leader", ""))
        if cand in names:
            votes[cand] = votes.get(cand, 0) + 1

    leader = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
    return {"leader": leader, "votes": votes}

def agg_matching(agent_jsons: List[Dict[str, Any]], G: nx.Graph) -> Dict[str, Any]:
    """
    agent_jsons: [{"node":"A","match":"B"}, ...]
    return: {"pairs":[("A","B"),...],"unmatched":[...]}
    """
    pairs = set()
    names = _nameset(G)
    for rec in agent_jsons:
        n1 = str(rec.get("node", ""))
        n2 = str(rec.get("match", ""))
        if n1 in names and n2 in names:
            # 双向去重：{A,B} 与 {B,A} 视为同一对
            key = tuple(sorted([n1, n2]))
            pairs.add(key)
    matched = {a for ab in pairs for a in ab}
    unmatched = list(names - matched)
    return {"pairs": sorted(list(pairs)), "unmatched": unmatched}

def agg_vertex_cover(agent_jsons: List[Dict[str, Any]], G: nx.Graph) -> Dict[str, Any]:
    """
    agent_jsons: [{"node":"A","cover":true}, ...]
    return: {"cover":[names], "is_valid":bool}
    """
    cover = set()
    names = _nameset(G)
    for rec in agent_jsons:
        n = str(rec.get("node", ""))
        if n not in names:
            continue
        flag = rec.get("cover", False)
        if isinstance(flag, str):
            flag = flag.lower() in ("true", "1", "yes", "y")
        if flag:
            cover.add(n)

    UG = _undirected(G)
    ok = all((_name(G,u) in cover) or (_name(G,v) in cover) for u,v in UG.edges())
    return {"cover": sorted(list(cover)), "is_valid": ok}

def aggregate(task_type: str, agent_jsons: List[Dict[str, Any]], G: nx.Graph) -> Dict[str, Any]:
    """统一接口，根据任务类型分派聚合函数"""
    task = task_type.lower()
    if task == "coloring":
        return agg_coloring(agent_jsons, G)
    elif task == "consensus":
        return agg_consensus(agent_jsons, G)
    elif task in ("leader", "leader_election"):
        return agg_leader_election(agent_jsons, G)
    elif task == "matching":
        return agg_matching(agent_jsons, G)
    elif task == "vertex_cover":
        return agg_vertex_cover(agent_jsons, G)
    else:
        return {"raw_outputs": agent_jsons, "note": f"No aggregator defined for {task_type}"}
