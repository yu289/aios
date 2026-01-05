from typing import Dict, Any, List, Callable
import json
from .schemas import TASK_SCHEMAS
from .aggregators import (
    agg_coloring, agg_consensus, agg_leader_election, agg_matching, agg_vertex_cover
)

AGGREGATORS: Dict[str, Callable] = {
    "coloring": agg_coloring,
    "consensus": agg_consensus,
    "leader_election": agg_leader_election,
    "matching": agg_matching,
    "vertex_cover": agg_vertex_cover,
}

def build_system_prompt(task: str, neighbor_names: List[str], rounds: int) -> str:
    """
    统一的“最终一轮仅输出 JSON”提示；中间轮仍按你原本的交互逻辑。
    """
    schema = TASK_SCHEMAS[task]["agent_final_schema"]
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    neigh = ", ".join(neighbor_names) if neighbor_names else "（无邻居）"
    return (
        "你是一个多智能体网络中的节点。你只能与邻居通信，在每一轮接收邻居的消息并发送你的消息。\n"
        f"邻居列表：{neigh}\n"
        f"总轮数：{rounds}。在第 {rounds} 轮结束时，你必须**只输出一个严格的 JSON**，不得包含额外文本。\n"
        "该 JSON 必须满足如下 Schema（最后一轮才输出）：\n"
        f"{schema_str}\n"
        "注意：在前几轮你可以用自然语言与邻居交换信息；但最后一轮必须只输出 JSON。"
    )

def parse_agent_final_json(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取 JSON（容忍模型前后有多余文字；只取第一个 {...}）。
    """
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("Agent final message is not valid JSON.")
    return json.loads(m.group(0))

def aggregate(task: str, per_agent_jsons: List[Dict[str, Any]], nx_graph) -> Dict[str, Any]:
    """
    按任务类型调用对应聚合器，产出最终业务答案。
    """
    if task not in AGGREGATORS:
        raise KeyError(f"No aggregator for task '{task}'.")
    return AGGREGATORS[task](per_agent_jsons, nx_graph)
