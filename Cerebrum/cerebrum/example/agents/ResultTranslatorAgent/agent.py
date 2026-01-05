import json
from cerebrum.llm.apis import llm_chat_with_json_output
import re

RESULT_SYSTEM_PROMPT = """You are a Result Explainer Agent.
Your job: read a task type and a JSON final_answer produced by a multi-agent system, then produce a clear, human-readable summary in Chinese.

Rules:
1) Output STRICT JSON only, no extra text, with this schema:
{
  "task": "<one of the five types>",
  "headline": "<one sentence Chinese summary>",
  "bullets": ["<brief bullet 1>", "<brief bullet 2>", "..."],
  "validity": "<OK / NOT_OK / UNKNOWN>",
  "notes": "<optional extra notes or empty string>"
}
2) Be concise and faithful. Chinese only.
"""

def _extract_json(s: str):
    m = re.search(r"\{.*\}", s or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

class ResultTranslatorAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.messages = []

    def _call_llm_json(self):
        try:
            return llm_chat_with_json_output(agent_name=self.agent_name, messages=self.messages)["response"]["response_message"]
        except TypeError:
            return llm_chat_with_json_output(messages=self.messages)["response"]["response_message"]

    def run(self, task_input):
        if isinstance(task_input, str):
            task_input = json.loads(task_input)

        task_type = task_input["task_type"]
        final_answer = task_input.get("final_answer", {}) or {}
        meta = task_input.get("meta", {}) or {}

        payload = {"task": task_type, "final_answer": final_answer, "meta": meta}

        self.messages = [
            {"role": "system", "content": RESULT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        raw = self._call_llm_json()
        obj = _extract_json(raw)

        # 确保返回的 JSON 结果包含 `result` 键
        if isinstance(obj, dict) and all(k in obj for k in ["task", "headline", "bullets", "validity", "notes"]):
            obj["result"] = "Translation successful"  # 确保返回 result 字段
            return {"agent_name": self.agent_name, **obj}

        # 本地兜底（如果解析失败）
        return {
            "agent_name": self.agent_name,
            "task": task_type,
            "headline": "结果摘要",
            "bullets": [f"{k}: {v}" for k, v in list(final_answer.items())[:6]] or ["（结果为空）"],
            "validity": "UNKNOWN",
            "notes": "（翻译失败，使用本地兜底）",
            "result": "Translation failed"  # 本地兜底时仍包含 `result` 键
        }
