#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import LiteralMessagePassing as lmp
import argparse
import asyncio
import datetime
import os
import json
import random
import logging
import re
import networkx as nx
from networkx.readwrite import json_graph
from utils import *
from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from typing import Dict, Any, List, Optional

from task_protocols import build_system_prompt as proto_build_prompt
from task_protocols import parse_agent_final_json as proto_parse_final
from task_protocols import aggregate as proto_aggregate

# ========== 基础设置 ==========
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

TASKS = {
    "matching": lmp.Matching,
    "consensus": lmp.Consensus,
    "coloring": lmp.Coloring,
    "leader_election": lmp.LeaderElection,
    "vertex_cover": lmp.VertexCover,
}

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL", "")

TASK_LABELS = ["consensus", "leader_election", "matching", "coloring", "vertex_cover"]

SYSTEM_PROMPT = """You are a routing ChatAgent. Classify the user's request into EXACTLY ONE of five task types:
- consensus: everyone should output the SAME single value/config/version.
- leader_election: exactly ONE leader/owner/gateway should be chosen.
- matching: pairwise ONE-TO-ONE matching with mutual confirmation.
- coloring: partition nodes so that ADJACENT nodes are in DIFFERENT groups (conflict avoidance).
- vertex_cover: choose a MINIMAL set of nodes that COVERS ALL edges/links.

Output STRICT JSON ONLY with this schema:
{"task_type": "<one-of: consensus|leader_election|matching|coloring|vertex_cover>", "reason": "<short english or chinese explanation>"}"""

RESULT_SYSTEM_PROMPT = """You are a Result Explainer Agent.
Your job: read a task type and a JSON final_answer produced by a multi-agent system, then produce a clear, human-readable summary in Chinese.

Task types (exactly these five):
- coloring
- consensus
- leader_election
- matching
- vertex_cover

Rules:
1) Output STRICT JSON only, no extra text, with this schema:
{
  "task": "<one of the five types>",
  "headline": "<one sentence Chinese summary>",
  "bullets": ["<brief bullet 1>", "<brief bullet 2>", "..."],
  "validity": "<OK / NOT_OK / UNKNOWN>",
  "notes": "<optional extra notes or empty string>"
}
2) Be concise and faithful to given final_answer JSON. Do not invent fields not present.
3) Chinese only. Keep bullets short (8~20 chars each).
4) If something is missing, set validity to UNKNOWN and explain shortly in notes.
"""

# 关键词兜底与优先级（仅用于任务分类兜底；与拓扑无关）
_KEYWORDS = {
    "leader_election": ["唯一负责人","唯一主","选举","竞选","leader","主节点","唯一网关","single coordinator"],
    "matching": ["配对","结对","双向确认","一对一","1-to-1","matching","撮合","绑定","pairing"],
    "coloring": ["相邻不同","相邻不得","互斥","冲突避免","邻接冲突","颜色","coloring","信道","干扰"],
    "vertex_cover": ["覆盖所有","覆盖全部","最少","最小监控","最小哨点","守护点","cover all edges","vertex cover"],
    "consensus": ["统一","一致","同一结果","口径一致","同一版本","same version","agree on the same","共识"],
}
_PRIORITY = ["leader_election", "matching", "coloring", "vertex_cover", "consensus"]

def _keyword_fallback(text: str):
    """返回(任务类型, 命中关键词列表)"""
    lower = text.lower()
    hits = []
    for t, kws in _KEYWORDS.items():
        for kw in kws:
            if kw in text or kw.lower() in lower:
                hits.append((t, kw))
                break
    for t in _PRIORITY:
        if any(h[0] == t for h in hits):
            return t, [kw for typ, kw in hits if typ == t]
    return "consensus", []  # 默认回退

def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def build_task_agent() -> ChatAgent:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=MODEL_NAME,
        api_key=API_KEY,
        url=API_BASE,
        model_config_dict={"response_format": {"type": "json_object"}},
    )
    return ChatAgent(system_message=SYSTEM_PROMPT, model=model)

def build_result_agent() -> ChatAgent:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=MODEL_NAME,
        api_key=API_KEY,
        url=API_BASE,
        model_config_dict={"response_format": {"type": "json_object"}},
    )
    return ChatAgent(system_message=RESULT_SYSTEM_PROMPT, model=model)

async def _agent_step(agent, text: str):
    resp = agent.step(text)
    if hasattr(resp, "__await__"):
        resp = await resp
    return resp  # ChatAgentResponse

async def classify_with_camel(task_text: str) -> dict:
    """
    返回结构：
    {
      "task_type": str,
      "source": "camel" | "keyword_fallback",
      "reason": str | None,
      "raw": str | None,
      "retries": int,
      "keyword_hits": [str, ...]
    }
    """
    agent = build_task_agent()

    async def _once():
        try:
            resp = await _agent_step(agent, task_text)
            raw = None
            if hasattr(resp, "msgs") and resp.msgs:
                raw = resp.msgs[0].content
            elif hasattr(resp, "content"):
                raw = resp.content
            elif isinstance(resp, str):
                raw = resp
            else:
                raw = str(resp)
            obj = _extract_json(raw or "")
            if obj and isinstance(obj, dict):
                t = obj.get("task_type", "")
                reason = obj.get("reason", None)
                if t in TASK_LABELS:
                    return {"task_type": t, "reason": reason, "raw": raw}
        except Exception as e:
            logger.error("CAMEL classify error: %r", e)
        return None

    info = await _once()
    retries = 0
    if info is None:
        await asyncio.sleep(0.5)
        info = await _once()
        retries = 1

    if info is None:
        # 关键词兜底
        t, kws = _keyword_fallback(task_text)
        result = {
            "task_type": t,
            "source": "keyword_fallback",
            "reason": f"关键词兜底命中：{', '.join(kws) if kws else '无明确命中，按默认优先级回退'}",
            "raw": None,
            "retries": retries,
            "keyword_hits": kws,
        }
    else:
        result = {
            "task_type": info["task_type"],
            "source": "camel",
            "reason": info.get("reason"),
            "raw": info.get("raw"),
            "retries": retries,
            "keyword_hits": [],
        }
    # 观测输出
    print("\n=== Task Classification ===")
    print(f"Source     : {result['source']}")
    print(f"Task Type  : {result['task_type']}")
    if result["reason"]:
        print(f"Reason     : {result['reason']}")
    if result["keyword_hits"]:
        print(f"KW Hits    : {result['keyword_hits']}")
    print(f"Retries    : {result['retries']}")
    if result["raw"] and isinstance(result["raw"], str):
        preview = (result["raw"][:200] + "...") if len(result["raw"]) > 200 else result["raw"]
        print(f"Raw (head) : {preview}")
    print("==========================\n")
    return result

async def explain_with_camel(task_type: str, final_answer: dict, extra_meta: dict | None = None) -> dict:
    """
    返回：
      {"task": "...", "headline": "...", "bullets": [...], "validity": "OK|NOT_OK|UNKNOWN", "notes": "..."}
    """
    agent = build_result_agent()

    user_payload = {
        "task": task_type,
        "final_answer": final_answer,
        "meta": extra_meta or {},
    }

    async def _once():
        try:
            resp = agent.step(json.dumps(user_payload, ensure_ascii=False))
            if hasattr(resp, "__await__"):
                resp = await resp
            raw = None
            if hasattr(resp, "msgs") and resp.msgs:
                raw = resp.msgs[0].content
            elif hasattr(resp, "content"):
                raw = resp.content
            elif isinstance(resp, str):
                raw = resp
            else:
                raw = str(resp)
            m = re.search(r"\{.*\}", raw or "", flags=re.S)
            if not m:
                return None, raw
            obj = json.loads(m.group(0))
            need = {"task","headline","bullets","validity","notes"}
            if isinstance(obj, dict) and need.issubset(obj.keys()):
                return obj, raw
        except Exception as e:
            return None, f"ERROR: {e}"
        return None, raw
    # 尝试 2 次
    parsed, raw = await _once()
    if parsed is None:
        await asyncio.sleep(0.3)
        parsed, raw = await _once()

    if parsed:
        print("\n=== Result Agent (CAMEL) ===")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        return parsed

    # ---- 兜底（不用模型也能给出可读解释）----
    def _fallback(task: str, ans: dict) -> dict:
        t = task.lower()
        head = "结果摘要"
        bullets = []
        validity = "UNKNOWN"
        notes = ""

        if t == "coloring":
            ch = ans.get("channels", {})
            ok = ans.get("valid")
            head = "图着色（信道分配）结果"
            bullets = [f"{k}→信道{v}" for k, v in ch.items()][:6]
            validity = "OK" if ok else "NOT_OK"
        elif t == "consensus":
            val = ans.get("value")
            agr = ans.get("agreement")
            head = "一致性结果"
            bullets = [f"一致值: {val}", f"达成一致: {bool(agr)}"]
            validity = "OK" if agr else "NOT_OK"
        elif t == "leader_election":
            leader = ans.get("leader")
            votes = ans.get("votes", {})
            head = "选主结果"
            bullets = [f"领导者: {leader}"] + [f"{k}:{v}票" for k, v in list(votes.items())[:5]]
            validity = "OK" if leader else "NOT_OK"
        elif t == "matching":
            pairs = ans.get("pairs", [])
            unmatched = ans.get("unmatched", [])
            head = "配对结果"
            bullets = [f"{a}↔{b}" for a,b in pairs[:6]]
            if unmatched:
                bullets.append("未匹配: " + "、".join(unmatched[:6]))
            validity = "OK"  # 是否稳定留给评分器，这里只做展示
        elif t == "vertex_cover":
            cover = ans.get("cover", [])
            ok = ans.get("is_valid")
            head = "顶点覆盖结果"
            bullets = ["覆盖: " + "、".join(cover[:8])]
            validity = "OK" if ok else "NOT_OK"
        else:
            head = f"{t} 任务结果"
            bullets = [f"{k}: {v}" for k,v in list(ans.items())[:6]]

        return {
            "task": t,
            "headline": head,
            "bullets": bullets or ["（结果为空）"],
            "validity": validity,
            "notes": "（CAMEL 解释失败，使用本地兜底）"
        }

    fb = _fallback(task_type, final_answer or {})
    print("\n=== Result Agent (Fallback) ===")
    print(json.dumps(fb, ensure_ascii=False, indent=2))
    return fb

def _scale(n: int) -> str:
    if n <= 8: return "small"
    if n <= 16: return "medium"
    return "large"

# === 使用 topology_priority.json ===
def _load_topology_cfg(path: str = "topology_priority.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"topology_priority.json not found at '{path}'. "
            "Please place the config file next to main.py."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 规范化任务名和拓扑名为小写；保留 guards
    tasks_norm: Dict[str, Any] = {}
    for task_name, bucket in raw.get("tasks", {}).items():
        tkey = task_name.strip().lower()
        tasks_norm[tkey] = {}
        for size in ["small", "medium", "large"]:
            if size in bucket:
                tasks_norm[tkey][size] = [x.strip().lower() for x in bucket[size]]
            else:
                tasks_norm[tkey][size] = []
        if "guards" in bucket:
            tasks_norm[tkey]["_guards"] = bucket["guards"]

    if not tasks_norm:
        raise ValueError("Invalid topology_priority.json: 'tasks' is empty or malformed.")

    return {"tasks": tasks_norm, "raw": raw}

def _rank_topologies(task_name: str, n_agents: int, cfg_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    size = _scale(n_agents)
    task_name = task_name.strip().lower()
    if task_name not in cfg_json["tasks"]:
        raise KeyError(f"Task '{task_name}' not configured in topology_priority.json.")

    bucket = cfg_json["tasks"][task_name]
    base_list = list(bucket.get(size, []))
    if not base_list:
        raise KeyError(f"No topology list for task '{task_name}' at size '{size}' in topology_priority.json.")

    # 初始分 [0.6,0.9]
    base_rank = {name: i for i, name in enumerate(base_list)}
    denom = max(1, len(base_list) - 1)
    scores = {name: 0.9 - 0.3 * (base_rank[name] / denom) for name in base_list}
    why = {name: [f"Base priority({task_name}, {size}) rank={base_rank[name]+1}"] for name in base_list}

    # 应用 guards（ban/penalize/prefer）
    guards = bucket.get("_guards", [])
    for g in guards:
        ok_sizes = g.get("when", {}).get("size", "")
        if ok_sizes and size not in ok_sizes.split("|"):
            continue
        for t in g.get("ban", []):
            tl = t.lower()
            if tl in scores:
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

def select_topology(task_name: str, n_agents: int, cfg_json: Dict[str, Any]) -> str:
    ranked = _rank_topologies(task_name, n_agents, cfg_json)
    topo = ranked[0]["name"]
    print(f"[SelectTopology] task={task_name} scale={_scale(n_agents)} -> topology={topo}")
    for i, cand in enumerate(ranked, 1):
        print(f"  - #{i} {cand['name']} (score={cand['score']}) :: {cand['why']}")
    return topo

MODEL_PROVIDER = {
    "gpt-4.1": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4.1-nano": "openai",
    "o1": "openai",
    "o3": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",
    "glm-4-plus": "openai",
    "qwen3-32b-awq": "openai",
    "Qwen2.5-72B-Int4": "openai",
    "llama3.1": "ollama",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-7-sonnet-20250219-thinking": "anthropic",
    "gemini-1.5-pro": "google-genai",
    "gemini-2.0-flash": "google-genai",
    "gemini-2.0-flash-lite": "google-genai",
    "gemini-2.5-pro-preview-05-06": "google-genai",
}

GEN_MAP = {
    "chain": generate_chain_graph,
    "star": generate_star_graph,
    "tree": generate_tree_graph,
    "net": generate_net_graph,
    "mlp": generate_mlp_graph,
    "random": generate_random_dag,
}

def save_results(answers, transcripts, graph, rounds, model_name, task, score,
                 graph_generator, graph_index, successful, error_message,
                 chain_of_thought, num_fallbacks, num_failed_json_parsings_after_retry,
                 num_failed_answer_parsings_after_retry, final_answer=None):
    """Saves the experiment results and message transcripts to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task}_results_{timestamp}_rounds{rounds}_{model_name.split('/')[-1]}_nodes{len(graph.nodes())}.json"

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    def _undirected_for_stats(G: nx.Graph) -> nx.Graph:
        UG = G.to_undirected() if G.is_directed() else G
        if UG.number_of_nodes() == 0:
            return UG
        if not nx.is_connected(UG):
            largest_cc = max(nx.connected_components(UG), key=len)
            UG = UG.subgraph(largest_cc).copy()
        return UG

    UG = _undirected_for_stats(graph)
    diameter = int(nx.diameter(UG)) if UG.number_of_nodes() > 0 else 0
    max_degree = int(max(dict(UG.degree()).values())) if UG.number_of_nodes() > 0 else 0

    stats = {
        'answers': answers,
        'transcripts': transcripts,
        'graph': json_graph.node_link_data(graph),
        'num_nodes': len(graph.nodes()),
        'diameter': diameter,
        'max_degree': max_degree,
        'rounds': rounds,
        'model_name': model_name,
        'task': task,
        'score': score,
        'graph_generator': graph_generator,
        'graph_index': graph_index,
        'successful': successful,
        'error_message': error_message,
        'chain_of_thought': chain_of_thought,
        'num_fallbacks': num_fallbacks,
        'num_failed_json_parsings_after_retry': num_failed_json_parsings_after_retry,
        'num_failed_answer_parsings_after_retry': num_failed_answer_parsings_after_retry,
        'directed': bool(graph.is_directed()),
        'final_answer': final_answer,
    }
    if graph.is_directed():
        stats['max_in_degree'] = int(max(dict(graph.in_degree()).values())) if graph.number_of_nodes() > 0 else 0
        stats['max_out_degree'] = int(max(dict(graph.out_degree()).values())) if graph.number_of_nodes() > 0 else 0

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"[Results] Saved to {filepath}")

def determine_rounds(task, graph, num_sample, num_samples, rounds):
    UG = graph.to_undirected() if graph.is_directed() else graph
    if UG.number_of_nodes() == 0:
        return rounds
    if not nx.is_connected(UG):
        largest_cc = max(nx.connected_components(UG), key=len)
        UG = UG.subgraph(largest_cc).copy()
    d = nx.diameter(UG)
    if task in ["consensus", "leader_election"] or graph.number_of_nodes() > 16:
        r = 2 * d + 1
    else:
        r = rounds
    print(f"[Rounds] task={task} nodes={graph.number_of_nodes()} diam={d} -> rounds={r}")
    return r

async def run(args):
    """
    执行流程：
      - 任务来源：--task（直指） 或 --task_text（CAMEL 判定，失败走关键词兜底）
      - 严格按 topology_priority.json 选拓扑 -> 造图 -> 估计回合 -> 执行与计分 -> 保存结果
    """
    import traceback

    n_agents: int = args.n_agents
    default_rounds = getattr(args, "rounds", 4)
    seed = getattr(args, "seed", 42)
    chain_of_thought = not getattr(args, "disable_chain_of_thought", False)

    model_name = getattr(args, "model", MODEL_NAME) or "qwen3-32b-awq"
    model_provider = MODEL_PROVIDER.get(model_name, "openai")
    print(f"[Model] name={model_name} provider={model_provider}")

    # 任务类型：直接指定 or 自然语言分类
    if getattr(args, "task", None):
        task_name: str = args.task
        print(f"[Task] Directly specified: {task_name}")
    else:
        task_text = getattr(args, "task_text", None)
        if not task_text:
            raise ValueError("You must provide either --task (one of 5 types) or --task_text (natural language).")
        cls = await classify_with_camel(task_text)
        task_name = cls["task_type"]
        print(f"[Task] From {'CAMEL' if cls['source']=='camel' else 'Keyword Fallback'} -> {task_name}")

    random.seed(seed)

    # 加载拓扑优先级配置（强制要求存在）
    topo_cfg = _load_topology_cfg("topology_priority.json")

    # 选择拓扑（严格来自配置）
    graph_model: str = select_topology(task_name, n_agents, topo_cfg)
    if graph_model not in GEN_MAP:
        raise ValueError(f"Unknown graph_model '{graph_model}'. Available: {list(GEN_MAP.keys())}")

    gen_fn = GEN_MAP[graph_model]
    try:
        graph = gen_fn(n_agents, seed=seed, directed=True)
    except TypeError:
        graph = gen_fn(n_agents)

    graph = relabel_and_name_vertices(graph)
    print(f"[Graph] generator={graph_model} nodes={graph.number_of_nodes()} directed={graph.is_directed()}")

    graph_index = 0
    rounds = determine_rounds(task_name, graph, graph_index, 1, default_rounds)

    task_class = TASKS[task_name]
    lmp_model: lmp.LiteralMessagePassing = task_class(
        graph=graph,
        rounds=rounds,
        model_name=model_name,
        model_provider=model_provider,
        chain_of_thought=chain_of_thought,
        protocol_enabled=True,
        protocol_task_name=task_name,              # "coloring"/"consensus"/...
        protocol_build_prompt=lambda neighs, r: proto_build_prompt(task_name, neighs, r),
        protocol_parse_final=proto_parse_final,
        protocol_aggregate=lambda agent_jsons, G: proto_aggregate(task_name, agent_jsons, G),
    )

    print("[LMP] Bootstrapping...")
    await lmp_model.bootstrap()

    try:
        print("[LMP] Passing messages...")
        answers = await lmp_model.pass_messages()
        score = lmp_model.get_score(answers)
        successful = True
        error_message = None
        print(f"[Score] task={task_name} score={score}")
    except Exception as e:
        answers = [None for _ in range(graph.order())]
        score = None
        successful = False
        # error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        error_message = f"{type(e).__name__}: {e}"
        print(f"[Error] {error_message}")

    final_answer = None
    try:
        final_answer = lmp_model.get_final_answer()   # <- 新增
        if final_answer is not None:
            print("[Final Answer]", json.dumps(final_answer, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[Final Answer] aggregation failed: {e}")
    save_results(
        answers=answers,
        transcripts=lmp_model.get_transcripts(),
        graph=lmp_model.graph,
        rounds=rounds,
        model_name=lmp_model.model_name,
        task=task_name,
        score=score,
        graph_generator=graph_model,
        graph_index=graph_index,
        successful=successful,
        error_message=error_message,
        chain_of_thought=chain_of_thought,
        num_fallbacks=getattr(lmp_model, "num_fallbacks", None),
        num_failed_json_parsings_after_retry=getattr(lmp_model, "num_failed_json_parsings_after_retry", None),
        num_failed_answer_parsings_after_retry=getattr(lmp_model, "num_failed_answer_parsings_after_retry", None),
        final_answer=final_answer
    )
    # 将最终业务答案写入结果文件（附加）
    try:
        # 你 save_results 里生成的文件路径规则可以复用；这里简单再写一份 sidecar
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        sidecar = os.path.join(out_dir, "final_answer_latest.json")
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "final_answer": final_answer,
            }, f, ensure_ascii=False, indent=2)
        print(f"[Final Answer] saved to {sidecar}")
    except Exception as e:
        print(f"[Final Answer] save failed: {e}")
        # === 使用 Result Agent 产出“人话”解释并落盘 ===
    try:
        result_expl = None
        if final_answer is not None:
            # 附带一些元信息（可选）
            meta = {
                "num_nodes": graph.number_of_nodes(),
                "rounds": rounds,
                "topology": graph_model,
                "score": score,
            }
            result_expl = await explain_with_camel(task_name, final_answer, meta)

            # 控制台打印
            print("\n=== Result Agent (Readable) ===")
            print(result_expl.get("headline", ""))
            for b in result_expl.get("bullets", []):
                print(" - " + b)
            if "validity" in result_expl:
                print("  结论: " + result_expl["validity"])
            if result_expl.get("notes"):
                print("  说明: " + result_expl["notes"])

            # 保存到文本
            out_dir = "results"
            os.makedirs(out_dir, exist_ok=True)
            txt_path = os.path.join(out_dir, "result_latest.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task_name}\n")
                f.write(result_expl.get("headline","") + "\n")
                for b in result_expl.get("bullets", []):
                    f.write(" - " + b + "\n")
                if "validity" in result_expl:
                    f.write("结论: " + result_expl["validity"] + "\n")
                if result_expl.get("notes"):
                    f.write("说明: " + result_expl["notes"] + "\n")
            print(f"[ResultAgent] saved to {txt_path}")
    except Exception as e:
        print(f"[ResultAgent] Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=TASK_LABELS, help="直接指定五类任务之一")
    parser.add_argument("--task_text", type=str, help="自然语言描述；用 CAMEL ChatAgent 自动分类")
    parser.add_argument("--n_agents", type=int, required=True, help="智能体数量")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_chain_of_thought", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="覆盖默认模型")
    args = parser.parse_args()
    asyncio.run(run(args))
