import os, json, re
from cerebrum.llm.apis import llm_chat_with_json_output

TASK_LABELS = ["consensus", "leader_election", "matching", "coloring", "vertex_cover"]

# 你 main.py 的 SYSTEM_PROMPT 逻辑（建议直接放 config["description"]，这里保底）
SYSTEM_PROMPT = """You are a routing ChatAgent. Classify the user's request into EXACTLY ONE of five task types:
- consensus: everyone should output the SAME single value/config/version.
- leader_election: exactly ONE leader/owner/gateway should be chosen.
- matching: pairwise ONE-TO-ONE matching with mutual confirmation.
- coloring: partition nodes so that ADJACENT nodes are in DIFFERENT groups (conflict avoidance).
- vertex_cover: choose a MINIMAL set of nodes that COVERS ALL edges/links.

Output STRICT JSON ONLY with this schema:
{"task_type":"<one-of: consensus|leader_election|matching|coloring|vertex_cover>","reason":"<short english or chinese explanation>"}"""

_KEYWORDS = {
    "leader_election": ["唯一负责人","选举","leader","主节点","唯一网关"],
    "matching": ["配对","结对","一对一","matching","pairing"],
    "coloring": ["相邻不同","互斥","冲突避免","颜色","coloring"],
    "vertex_cover": ["覆盖所有","最少","最小","vertex cover","cover all edges"],
    "consensus": ["统一","一致","同一结果","共识","agree on the same"],
}
_PRIORITY = ["leader_election", "matching", "coloring", "vertex_cover", "consensus"]

def _keyword_fallback(text: str):
    lower = (text or "").lower()
    hits = []
    for t, kws in _KEYWORDS.items():
        for kw in kws:
            if kw in (text or "") or kw.lower() in lower:
                hits.append((t, kw))
                break
    for t in _PRIORITY:
        if any(h[0] == t for h in hits):
            return t, [kw for typ, kw in hits if typ == t]
    return "consensus", []

def _extract_json(s: str):
    m = re.search(r"\{.*\}", s or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _scale(n: int) -> str:
    if n <= 8: return "small"
    if n <= 16: return "medium"
    return "large"

def _load_topology_cfg(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks_norm = {}
    for task_name, bucket in raw.get("tasks", {}).items():
        tkey = task_name.strip().lower()
        tasks_norm[tkey] = {}
        for size in ["small", "medium", "large"]:
            tasks_norm[tkey][size] = [x.strip().lower() for x in bucket.get(size, [])]
        if "guards" in bucket:
            tasks_norm[tkey]["_guards"] = bucket["guards"]
    return {"tasks": tasks_norm, "raw": raw}

def _rank_topologies(task_name: str, n_agents: int, cfg_json):
    size = _scale(n_agents)
    task_name = task_name.strip().lower()
    bucket = cfg_json["tasks"][task_name]
    base_list = list(bucket.get(size, []))
    base_rank = {name: i for i, name in enumerate(base_list)}
    denom = max(1, len(base_list) - 1)
    scores = {name: 0.9 - 0.3 * (base_rank[name] / denom) for name in base_list}
    why = {name: [f"Base priority({task_name}, {size}) rank={base_rank[name]+1}"] for name in base_list}

    for g in bucket.get("_guards", []):
        ok_sizes = g.get("when", {}).get("size", "")
        if ok_sizes and size not in ok_sizes.split("|"):
            continue
        for t in g.get("ban", []):
            tl = t.lower()
            scores.pop(tl, None); why.pop(tl, None)
        for t in g.get("penalize", []):
            tl = t.lower()
            if tl in scores:
                scores[tl] -= 0.15; why[tl].append("penalize by guard")
        for t in g.get("prefer", []):
            tl = t.lower()
            if tl in scores:
                scores[tl] += 0.08; why[tl].append("prefer by guard")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"name": name, "score": round(sc,3), "why": "; ".join(why[name])} for name, sc in ranked]

def select_topology(task_name: str, n_agents: int, cfg_json):
    ranked = _rank_topologies(task_name, n_agents, cfg_json)
    return ranked[0]["name"], ranked

class TaskClassifierAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.messages = []

    def _call_llm_json(self):
        # 兼容你遇到的签名差异：优先 agent_name 必传
        try:
            return llm_chat_with_json_output(agent_name=self.agent_name, messages=self.messages)["response"]["response_message"]
        except TypeError:
            return llm_chat_with_json_output(messages=self.messages)["response"]["response_message"]

    def run(self, task_input):
        """
        task_input 支持：
        1) dict: {"task_text":"...", "n_agents": 12}
        2) str(json): 同上
        """
        if isinstance(task_input, str):
            try:
                task_input = json.loads(task_input)
            except Exception:
                task_input = {"task_text": task_input, "n_agents": None}

        task_text = (task_input.get("task_text") or "").strip()
        n_agents = task_input.get("n_agents")

        if not task_text or not isinstance(n_agents, int) or n_agents <= 0:
            return {"agent_name": self.agent_name, "result": "输入必须包含 task_text 和 n_agents(>0)"}

        # 1) 任务分类（五选一）
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": task_text}]
        raw = self._call_llm_json()
        obj = _extract_json(raw)
        if not (isinstance(obj, dict) and obj.get("task_type") in TASK_LABELS):
            t, kws = _keyword_fallback(task_text)
            task_type = t
            reason = f"关键词兜底: {','.join(kws) if kws else '默认回退'}"
            source = "keyword_fallback"
        else:
            task_type = obj["task_type"]
            reason = obj.get("reason", "")
            source = "llm"

        # 2) 拓扑匹配（严格按 topology_priority.json）
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here,"topology_priority.json")
        # cfg_path = os.path.join("/work/Cerebrum/cerebrum/example/topo", "topology_priority.json")
        cfg_path = os.path.abspath(cfg_path)
        topo_cfg = _load_topology_cfg(cfg_path)

        topology, ranked = select_topology(task_type, n_agents, topo_cfg)
        # 为了避免你之前单测“reason 精确匹配”那种脆弱断言：
        # 给一个稳定短标签，同时保留 LLM 原因
        short_reason_map = {
            "consensus": "一致性",
            "leader_election": "选主",
            "matching": "匹配",
            "coloring": "着色",
            "vertex_cover": "点覆盖",
        }

        return {
            "agent_name": self.agent_name,
            "task_text": task_text,
            "n_agents": n_agents,
            "task_type": task_type,
            "reason": short_reason_map.get(task_type, task_type),   # 稳定短原因
            "raw_reason": reason,                                   # 原始原因
            "source": source,
            "scale": _scale(n_agents),
            "topology": topology,
            "ranked_topologies": ranked,
        }
