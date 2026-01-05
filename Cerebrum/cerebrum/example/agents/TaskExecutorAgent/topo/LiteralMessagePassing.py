from abc import ABC
import asyncio
import networkx as nx
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import json
import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import messages_to_dict
import re
import regex
from typing import Callable, Optional, List, Dict, Any
from .utils import names
import yaml
from cerebrum.llm.apis import llm_chat
from cerebrum.config.config_manager import config as aios_config

def _json_guard(neigh_names: list[str]) -> str:
    if not neigh_names:
        valid = "(no neighbors)"
    else:
        valid = ", ".join(neigh_names)
    return (
        "\n\nIMPORTANT:\n"
        f"- Return ONLY a valid JSON object (no code fences, no prose, no comments).\n"
        f"- Valid keys are exactly these neighbor names: {valid}\n"
        "- Values must be plain text strings.\n"
        "- If you do not want to message a neighbor, omit that key.\n"
    )

def parse_messages(response: str) -> dict[str, str] | None:
    """从模型文本中提取一个 {neighbor_name: message} 的 JSON 对象。
    - 自动剥离 ```json 代码块；
    - 允许文本里有其它内容，但只接受其中的第一个合法 JSON 对象；
    - 键必须在全局名字表（utils.names）里，值必须是字符串。
    """
    if response is None:
        return None
    text = response.replace("\\n", "\n").strip()
    fence = regex.findall(r"```(?:json)?\s*([\s\S]*?)```", text, regex.IGNORECASE)
    if fence:
        text = fence[-1].strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and all(k in names for k in obj.keys()) and all(isinstance(v, str) for v in obj.values()):
                return obj
        except json.JSONDecodeError:
            pass
    matches = regex.findall(r"(\{[\s\S]*?\})", text, regex.DOTALL, overlapped=True)
    for cand in matches:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and all(k in names for k in obj.keys()) and all(isinstance(v, str) for v in obj.values()):
                return obj
        except json.JSONDecodeError:
            continue
    return None


RATE_LIMITER_KWARGS = {
    # --- 兼容模型 ---
    "gpt-4.1": dict(
        requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=20
    ),
    "gpt-4.1-nano": dict(
        requests_per_second=20, check_every_n_seconds=0.1, max_bucket_size=40
    ),
    "o3": dict(
        requests_per_second=1, check_every_n_seconds=0.1, max_bucket_size=5
    ),
    "DMXAPI-HuoShan-DeepSeek-R1-671B-64k": dict(
        requests_per_second=1, check_every_n_seconds=0.2, max_bucket_size=2
    ),
    "DMXAPI-HuoShan-DeepSeek-V3": dict(
        requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=6
    ),
    "qwen3-32b-awq": dict(
        requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=20
    ),
    "Qwen3-32B-FP8": dict(
        requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=20
    ),
    "GLM-4.5-Flash": dict(
        requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=20
    ),
    "Qwen2.5-72B-Int4": dict(
        requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=20,
    ),
    "gpt-4o-mini": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=200,
    ),
    "gpt-4.1-mini": dict(
        requests_per_second=5,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=4,
    ),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": dict(
        requests_per_second=50,
        check_every_n_seconds=0.1,
        max_bucket_size=100,
    ),
    "gpt-4o": dict(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "o1": dict(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "o1-mini": dict(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),

    "o3-mini": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "o4-mini": dict(
        requests_per_second=5,
        check_every_n_seconds=0.2,
        max_bucket_size=20,
    ),
    "llama3.1": dict(
        requests_per_second=20,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.0-flash": dict(
        requests_per_second=100,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.0-flash-lite": dict(
        requests_per_second=100,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-5-haiku-20241022": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-opus-20240229": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-7-sonnet-20250219": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-7-sonnet-20250219-thinking": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.5-pro-exp-03-25": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    ),
    "gemini-2.5-pro-preview-03-25": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    ),
    "gemini-2.5-pro-preview-05-06": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    ),
    "gemini-2.5-flash-preview-04-17": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.5-flash-preview-04-17-thinking": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-1.5-pro": dict(
        requests_per_second=100,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
}

class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            return yaml.safe_load(f)

    def get_openai_api_key(self):
        return self.config.get("api_keys", {}).get("openai",None)

    def get_model_config(self, model_name):
        # 从配置文件获取指定模型的配置
        for model in self.config.get('llms', {}).get('models', []):
            if model['name'] == model_name:
                return model
        return None

class LiteralMessagePassing(ABC):
    def __init__(
        self,
        graph,
        rounds: int = 4,
        model_name="Qwen3-32B-FP8",
        model_provider="openai",
        chain_of_thought=True,
        base_url=None,
        api_key=None,
        # ===== 新增：协议钩子（默认关闭，不影响旧行为） =====
        protocol_enabled: bool = False,
        protocol_task_name: str | None = None,
        # 构建“最后一轮仅输出JSON”的额外说明；签名：fn(neighbor_names: list[str], rounds: int) -> str
        protocol_build_prompt: Callable[[List[str], int], str] | None = None,
        # 解析最后一轮 JSON；签名：fn(text: str) -> dict
        protocol_parse_final: Callable[[str], Dict[str, Any]] | None = None,
        # 从 JSON 中提取与 get_score 兼容的“choice”字符串；签名：fn(json_obj: dict, valid: list[str]) -> str | None
        protocol_extract_choice: Callable[[Dict[str, Any], List[str]], str | None] | None = None,
        # （可选）聚合器：fn(per_agent_jsons: list[dict], nx_graph) -> dict
        protocol_aggregate: Callable[[List[Dict[str, Any]], Any], Dict[str, Any]] | None = None,

        agent_name_prefix: str = "task_executor_agent",
        use_aios_kernel: bool = True,

        **kwargs,
    ):
        self.graph = graph
        self.rounds = rounds
        self.model_name = model_name

        config_path = "/home/yu/work/AIOS/aios/config/config.yaml"
        config = ConfigManager(config_path)
        # 获取指定模型的配置
        model_config = config.get_model_config(model_name)
        # 如果模型配置中存在 'hostname'，将其作为 base_url
        base_url = model_config.get('hostname', None) if model_config else None
        # 如果没有找到 base_url，则设置为默认值或抛出异常
        if not base_url:
            raise ValueError(f"Model '{model_name}' does not have a valid base_url (hostname).")
        # 获取 API 密钥（假设 config.get_openai_api_key() 从 AIOS 配置文件读取）
        api_key = config.get_openai_api_key()
        if not api_key:
            raise ValueError(f"API key for model '{model_name}' is not set.")
        
        self.chain_of_thought = chain_of_thought
        # ===== 协议钩子保存 =====
        self._protocol_enabled = bool(protocol_enabled)
        self._protocol_task = protocol_task_name
        self._build_prompt_fn = protocol_build_prompt
        self._parse_final_fn = protocol_parse_final
        self._extract_choice_fn = protocol_extract_choice
        self._aggregate_fn = protocol_aggregate

        self.agent_name_prefix = agent_name_prefix
        self.aios_kernel_url = aios_config.get_kernel_url()  # kernel url（用于监控）
        self.use_aios_kernel = use_aios_kernel

        # 存储每个 agent 最后一轮解析得到的 JSON（与节点 index 对齐）
        self.final_agent_jsons: list[dict | None] = [None for _ in graph.nodes()]

        # ===== LangGraph 工作流（保持原有） =====
        rate_cfg = RATE_LIMITER_KWARGS.get(
            model_name, dict(requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=10)
        )
        rate_limiter = InMemoryRateLimiter(**rate_cfg)

        # 根据不同的模型提供者初始化模型
        if model_provider == "ollama":
            self.model = ChatOllama(model=model_name, base_url=base_url)
        else:
            chat_kwargs = {}

            # 兼容 claude 3.7 thinking 后缀
            if model_name.startswith("claude-3-7-sonnet"):
                if model_name.endswith("thinking"):
                    model_name = "-".join(model_name.split("-")[:-1])
                    chat_kwargs.update(
                        {
                            "max_tokens": 5000,
                            "thinking": {"type": "enabled", "budget_tokens": 2000},
                        }
                    )

            # 兼容 gemini thinking 预览名
            if model_name == "gemini-2.5-flash-preview-04-17-thinking":
                model_name = "gemini-2.5-flash-preview-04-17"
                chat_kwargs.update({"thinking_budget": 24000, "include_thoughts": True})

            init_kwargs = {
                "model_provider": model_provider,
                "rate_limiter": rate_limiter,
                **chat_kwargs,
            }
            if base_url or os.getenv("OPENAI_BASE_URL"):
                init_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
            if api_key or os.getenv("OPENAI_API_KEY"):
                init_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")

            if model_name == "o3" and "reasoning" not in init_kwargs:
                init_kwargs["reasoning"] = {"effort": "medium"}

            # Initialize ChatOpenAI with AIOS configuration
            self.model = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, **kwargs)
            
        # Initialize other models (depending on model_provider)
        self.model = init_chat_model(model_name, **init_kwargs).with_retry(stop_after_attempt=10)
        # ===== 模型初始化（保持你原有行为） =====
        # rate_cfg = RATE_LIMITER_KWARGS.get(
        #     model_name, dict(requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=10)
        # )
        # rate_limiter = InMemoryRateLimiter(**rate_cfg)

        # if model_provider == "ollama":
        #     self.model = ChatOllama(model=model_name, base_url=os.environ["OLLAMA_URI"])
        # else:
        #     chat_kwargs = {}

        #     # 兼容 claude 3.7 thinking 后缀
        #     if model_name.startswith("claude-3-7-sonnet"):
        #         if model_name.endswith("thinking"):
        #             model_name = "-".join(model_name.split("-")[:-1])
        #             chat_kwargs.update(
        #                 {
        #                     "max_tokens": 5000,
        #                     "thinking": {"type": "enabled", "budget_tokens": 2000},
        #                 }
        #             )
        #     # 兼容 gemini thinking 预览名
        #     if model_name == "gemini-2.5-flash-preview-04-17-thinking":
        #         model_name = "gemini-2.5-flash-preview-04-17"
        #         chat_kwargs.update({"thinking_budget": 24000, "include_thoughts": True})

        #     init_kwargs = {
        #         "model_provider": model_provider,
        #         "rate_limiter": rate_limiter,
        #         **chat_kwargs,
        #     }
        #     if base_url or os.getenv("OPENAI_BASE_URL"):
        #         init_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
        #     if api_key or os.getenv("OPENAI_API_KEY"):
        #         init_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
        #     if model_name == "o3" and "reasoning" not in init_kwargs:
        #         init_kwargs["reasoning"] = {"effort": "medium"}
        #     if self.model_provider == "openai":
        #         self.model = ChatOpenAI(model=model_name, api_key=api_key, base_url=self.base_url, **kwargs)
        # # 你可以根据其他模型提供者继续进行配置
        #     self.model = init_chat_model(model_name, **init_kwargs).with_retry(stop_after_attempt=10)

        # ===== LangGraph 工作流（保持原有） =====
        self.workflow = StateGraph(state_schema=MessagesState)
        self.transcripts = []

        # def call_model(state: MessagesState):
        #     response = self.model.invoke(state["messages"])
        #     return {"messages": response}
        def _lc_to_openai(messages):

            out = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    role = "system"
                elif isinstance(m, HumanMessage):
                    role = "user"
                else:
                    role = "assistant"
                out.append({"role": role, "content": m.content})
            return out

        def call_model(state: MessagesState, config=None):
            # 取出 thread_id（你在 ainvoke 里传了 thread_id=str(node_id)）
            tid = None
            if config and isinstance(config, dict):
                tid = (config.get("configurable") or {}).get("thread_id")
            tid = tid if tid is not None else "0"

            agent_name = f"{self.agent_name_prefix}/node_{tid}"

            # 走 AIOS kernel：产生 syscall，平台可监控
            resp = llm_chat(
                agent_name=agent_name,
                messages=_lc_to_openai(state["messages"]),
                base_url=self.aios_kernel_url,
            )["response"]["response_message"]
            
            return {"messages": AIMessage(content=resp)}


        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # ===== 运行态缓存 =====
        self.messages = {v: [] for v in graph.nodes()}
        self.chat_history = {v: [] for v in graph.nodes()}
        self.num_fallbacks = [0 for v in graph.nodes()]
        self.num_failed_json_parsings_after_retry = [0 for v in graph.nodes()]
        self.num_failed_answer_parsings_after_retry = [0 for v in graph.nodes()]

        # ===== 模板 =====
        self.bootstrap_template = PromptTemplate.from_template(
            """You are an agent that is connected with other agents (your neighbors), who you communicate with.
Your neighbors can in turn communicate with their neighbors and so forth. {short_task_description}
The rules are as follows:
1. There are {num_agents} agents in total. Everybody has a unique name. Your name is {own_name}.
2. You can only communicate with your immediate neighbors ({neighbors}). You cannot see or directly communicate with anyone else, unless information is relayed by intermediate agents.
3. You can exchange text-based messages with your neighbors in rounds. In each round, you will first receive the last messages sent by your neighbors and then be asked to generate your response messages which your neighbors receive in the next round. This process repeats for {num_rounds} rounds of message passing. Importantly, the process is synchronous: Every agent decides on which messages to send at the same time and sees the messages from other agents only in the next round.
4. Everybody (including you) decides what to share or request from neighbors. In every round, think step-by-step about the next set of messages you want to send. Output a JSON string that contains your response messages.
5. The messages you send to your neighbors MUST be a single valid JSON object (no code fences, no extra text). Example if your neighbors are Alan and Bob:
{{"Alan": "Message that will be sent to Alan.", "Bob": "Message that will be sent to Bob."}}
6. After {num_rounds} message passes, you have to solve the following task: {long_problem_description}"""
        )

        self.cot_prompt = (
            "Elaborate your chain of thought step-by-step first, then output the messages for your neighbors."
        )
        self.cot_prompt_final_prediction = (
            "\n\nElaborate your chain of thought step-by-step first, then answer the following: "
        )
        if not chain_of_thought:
            self.cot_prompt = ""
            self.cot_prompt_final_prediction = ""

        self.bootstrap_ask_for_first_messages = PromptTemplate.from_template(
            "{cot_prompt} Output your messages in JSON format as specified earlier."
        )
        # 旧的“最终枚举答案”提示（兼容 get_score）
        self.format_instructions = PromptTemplate.from_template(
            "{question} Format your answer exactly as a single token equal to one of the valid options. "
            "Do not include any other text. Valid options: {valid_answers}"
        )

    # -------- 统一邻居：DiGraph 用 前驱∪后继；Graph 用 neighbors() --------
    def neighbor_ids(self, node_id):
        if self.graph.is_directed():
            return sorted(set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id)))
        return list(self.graph.neighbors(node_id))

    def neighbor_names(self, node_id):
        return [str(self.graph.nodes[i]["name"]) for i in self.neighbor_ids(node_id)]

    # -------- Fallbacks --------
    async def fallback_json_request(self, node_id):
        self.num_fallbacks[node_id] += 1
        neigh_names = self.neighbor_names(node_id)
        user_message = HumanMessage(
            content="Your messages could not be parsed into JSON. Please correct and try again."
            + _json_guard(neigh_names)
        )
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({"messages": [user_message]}, config=config)
        return parse_messages(response["messages"][-1].content)

    async def fallback_answer_request(self, node_id):
        self.num_failed_answer_parsings_after_retry[node_id] += 1
        valid_answers = self.get_valid_answers(node_id)
        user_message = HumanMessage(
            content=self.format_instructions.format(
                question=self.question_for_prediction, valid_answers=", ".join(valid_answers)
            )
        )
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({"messages": [user_message]}, config=config)
        message = response["messages"][-1].content
        # 兼容 “协议模式 + choice” 的兜底提取
        if self._protocol_enabled and self._parse_final_fn:
            try:
                obj = self._safe_parse_json(message)
                if obj:
                    choice = self._extract_choice(obj, valid_answers)
                    if choice:
                        return choice
            except Exception:
                pass
        return self.parse_answer(node_id, message)

    # -------- 解析工具 --------
    def parse_answer(self, node_id, message) -> str | None:
        """旧的最终答案枚举解析（兼容 get_score）"""
        valid_answers = self.get_valid_answers(node_id)
        pattern = r"(" + "|".join(re.escape(ans) for ans in valid_answers) + r")"
        parsed_answer = re.search(pattern, message)
        if parsed_answer:
            return parsed_answer.group(1)
        return None

    def _safe_parse_json(self, text: str) -> dict | None:
        if text is None:
            return None
        s = text.replace("\\n", "\n").strip()
        # 剥离代码块
        fence = regex.findall(r"```(?:json)?\s*([\s\S]*?)```", s, regex.IGNORECASE)
        if fence:
            s = fence[-1].strip()
        # 匹配第一个 {...}
        m = regex.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _extract_choice(self, obj: dict, valid_answers: list[str]) -> str | None:
        """优先从 JSON 里提取 choice；如未提供自定义提取器，则尝试常用字段。"""
        if self._extract_choice_fn:
            try:
                c = self._extract_choice_fn(obj, valid_answers)
                if c in valid_answers:
                    return c
            except Exception:
                pass
        # 通用兜底：choice / answer / value / leader / channel（转成 Group N）
        for k in ["choice", "answer", "value"]:
            v = obj.get(k)
            if isinstance(v, str) and v in valid_answers:
                return v
        # 任务常见字段 → 映射到 valid
        if "leader" in obj and isinstance(obj["leader"], str):
            v = obj["leader"]
            if v in valid_answers:
                return v
        if "channel" in obj:
            ch = obj["channel"]
            try:
                ch = int(ch)
                candidate = f"Group {ch}"
                if candidate in valid_answers:
                    return candidate
            except Exception:
                pass
        if "partner" in obj and (obj["partner"] is None or isinstance(obj["partner"], str)):
            v = obj["partner"] or "None"
            if v in valid_answers:
                return v
        if "in_cover" in obj:
            v = "Yes" if bool(obj["in_cover"]) else "No"
            if v in valid_answers:
                return v
        return None

    def _final_json_instruction(self, node_id: int, rounds_left: int) -> str:
        """若启用协议，则在最后一轮附加“只输出JSON”的硬性约束；包含 choice 兼容字段。"""
        if not self._protocol_enabled:
            return ""
        if rounds_left != 0:
            return ""
        neighs = self.neighbor_names(node_id)
        valid = ", ".join(self.get_valid_answers(node_id))
        # 自定义 builder 优先
        if self._build_prompt_fn:
            try:
                return "\n\n" + self._build_prompt_fn(neighs, self.rounds)
            except Exception:
                pass
        # 通用说明（包含 choice 字段）
        return (
            "\n\nIMPORTANT (FINAL JSON ONLY):\n"
            "- At the end of THIS round, output ONLY a single JSON object and NOTHING else.\n"
            f"- The object MUST include a string field \"choice\" whose value is EXACTLY one of: {valid}.\n"
            "- You MAY include extra fields that describe your decision (e.g., channel/decided/leader/partner/in_cover), "
            "but do not include any non-JSON text."
        )

    async def parse_response_to_dict(self, message_dict, node_id, last_round=False):
        last_message = message_dict[-1]["data"]["content"]
        if not last_round:
            messages_sent = parse_messages(last_message)
        else:
            # 协议模式：先解析 JSON，缓存；再抽取 choice 以兼容 get_score
            messages_sent = None
            if self._protocol_enabled:
                obj = self._safe_parse_json(last_message)
                if obj is None and self._parse_final_fn:
                    try:
                        obj = self._parse_final_fn(last_message)
                    except Exception:
                        obj = None
                if obj is not None:
                    self.final_agent_jsons[node_id] = obj
                    choice = self._extract_choice(obj, self.get_valid_answers(node_id))
                    if choice:
                        messages_sent = choice
            # 若未启用协议或未能提取 choice，回退到老的字符串解析
            if messages_sent is None:
                messages_sent = self.parse_answer(node_id, last_message)

        if messages_sent is None:
            if not last_round:
                return None, self.fallback_json_request(node_id)
            else:
                return None, self.fallback_answer_request(node_id)
        return messages_sent, None

    async def update_messages(self, results):
        fallback_tasks = []
        fallback_nodes = []
        for node, result in zip(self.graph.nodes(), results):
            message, fallback = result
            if message is None:
                fallback_nodes.append(node)
                fallback_tasks.append(fallback)
            else:
                self.messages[node] = message

        if len(fallback_tasks) > 0:
            messages = await asyncio.gather(*fallback_tasks)
            for node, message in zip(fallback_nodes, messages):
                self.messages[node] = message

    async def bootstrap(self):
        """Bootstrap all nodes asynchronously."""
        tasks = [self.bootstrap_node(v) for v in self.graph.nodes()]
        results = await asyncio.gather(*tasks)
        await self.update_messages(results)

    async def bootstrap_node(self, node_id):
        """Bootstraps node with task-specific instructions."""
        bootstrap_parameters = self.get_bootstrap_parameters(node_id)
        system_message = SystemMessage(content=self.bootstrap_template.format(**bootstrap_parameters))
        guard = _json_guard(self.neighbor_names(node_id))
        user_message = HumanMessage(content=self.bootstrap_ask_for_first_messages.format(cot_prompt=self.cot_prompt) + guard)

        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({"messages": [system_message, user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response["messages"])
        return await self.parse_response_to_dict(self.chat_history[node_id], node_id)

    def get_bootstrap_parameters(self, node_id):
        neighbors = ", ".join(self.neighbor_names(node_id))
        return {
            "short_task_description": self.short_task_description,
            "long_problem_description": self.long_problem_description,
            "num_agents": self.graph.order(),
            "own_name": self.graph.nodes[node_id]["name"],
            "neighbors": neighbors,
            "num_rounds": self.rounds,
        }

    async def message_passing(self, node_id: int, rounds_left: int, messages: dict[str, str]):
        """Handles message exchange between nodes."""
        last_round = rounds_left == 0
        if last_round:
            messages_str = "Message passing has finished, here are the last messages you got from your neighbors:\n\n"
        else:
            messages_str = "These are the messages from your neighbors:\n\n"
        for name, message in messages.items():
            messages_str += f"Message from {name}:\n\n{message}\n\n"

        neighbour_names = self.neighbor_names(node_id)
        silent_neighbors = [name for name in neighbour_names if name not in messages]
        if len(silent_neighbors) > 0:
            messages_str += (
                "The following neighbors did not send you a messages in this round: "
                + ", ".join(silent_neighbors)
                + "\n\n"
            )

        if not last_round:
            neighbors = ", ".join(neighbour_names)
            messages_str += (
                f"{self.cot_prompt} Output your messages in JSON for your neighbors. "
                f"You have {rounds_left} rounds of communication left before you need to decide. "
                f"Your neighbors are: {neighbors} "
            )
            if rounds_left == 1:
                messages_str += "These are the last messages that your neighbors will receive from you."
            messages_str += _json_guard(neighbour_names)
        else:
            # 旧的“最终答案枚举”提示（兼容 get_score）
            messages_str += self.cot_prompt_final_prediction + self.format_instructions.format(
                question=self.question_for_prediction, valid_answers=", ".join(self.get_valid_answers(node_id))
            )
            # 协议模式：追加“只输出 JSON”的硬约束（含 choice 字段要求）
            extra = self._final_json_instruction(node_id, rounds_left)
            if extra:
                messages_str += extra

        user_message = HumanMessage(content=messages_str)
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({"messages": [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response["messages"])
        return await self.parse_response_to_dict(self.chat_history[node_id], node_id, last_round)

    async def pass_messages(self):
        """Executes synchronous message passing rounds."""
        for round in range(1, self.rounds + 1):
            rounds_left = self.rounds - round
            all_messages_sent = {}
            for v in self.graph.nodes():
                all_messages_sent[v] = self.messages[v]

            tasks = []
            for v in self.graph.nodes():
                name = self.graph.nodes[v]["name"]
                messages_received = {}
                for neigh in self.neighbor_ids(v):
                    if name in all_messages_sent[neigh]:
                        neigh_name = self.graph.nodes[neigh]["name"]
                        messages_received[neigh_name] = all_messages_sent[neigh][name]
                tasks.append(self.message_passing(node_id=v, rounds_left=rounds_left, messages=messages_received))

            results = await asyncio.gather(*tasks)
            await self.update_messages(results)

        return self.get_final_answers(self.messages, len(self.graph.nodes))

    def get_final_answers(self, last_messages, num_nodes):
        answers = [None] * num_nodes
        for node in last_messages.keys():
            answers[node] = last_messages[node]
        return answers

    # ===== 新增：获取每个 agent 的最终 JSON（仅在协议模式下有值） =====
    def get_final_agent_jsons(self) -> list[dict | None]:
        return self.final_agent_jsons

    # ===== （可选）直接在类内做聚合，若你传入了聚合器 =====
    def get_final_answer(self):
        if self._protocol_enabled and self._aggregate_fn and any(self.final_agent_jsons):
            return self._aggregate_fn(self.final_agent_jsons, self.graph)
        return None

    def get_transcripts(self):
        """Returns the full transcript of all messages exchanged."""
        return {self.graph.nodes[v]["name"]: self.chat_history[v] for v in self.graph.nodes()}

    def get_score(self, answers: list[str]) -> float:
        raise NotImplementedError

    def get_valid_answers(self, node_id):
        raise NotImplementedError

class Consensus(LiteralMessagePassing):
    """Implementation for consensus without malicious parties."""
    def __init__(self, graph, rounds, model_name="Qwen3-32B-FP8", model_provider="openai", chain_of_thought=True, **kwargs):
    # def __init__(self, graph, rounds, model_name="GLM-4.5-Flash", model_provider="openai", chain_of_thought=True, **kwargs):

        super().__init__(graph, rounds, model_name, model_provider, chain_of_thought, **kwargs)

        self.rounds = rounds
        self.short_task_description = "Your goal is to agree on a single value with the other agents. The possible values that you can decide on are either 0 or 1."
        self.long_problem_description = (
            "After the last round, each agent must decide on a single value."
        )
        self.question_for_prediction = "Please answer with the value you agreed on."

        # super().__init__(graph, model_name, model_provider, chain_of_thought)

    def get_score(self, answers: list[str]) -> float:
        valid_values = {"0", "1"}
        if not all(x in valid_values for x in answers) or (
                answers.count('0') != self.graph.order() and answers.count('0') != 0):
            return 0.0
        else:
            return 1.0
        
    def get_valid_answers(self, node_id):
        return ["0", "1"]


class LeaderElection(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="Qwen3-32B-FP8", model_provider="openai", chain_of_thought=True, **kwargs):
    # def __init__(self, graph, rounds, model_name="GLM-4.5-Flash", model_provider="openai", chain_of_thought=True, **kwargs):

        super().__init__(graph, rounds, model_name, model_provider, chain_of_thought, **kwargs)

        self.rounds = rounds
        self.short_task_description = "Your task is to collaboratively solve the problem of electing a single leader."
        self.long_problem_description = (
            "You will be requested to state whether or not you are the leader. The response should either be 'Yes' or 'No'. "
            "The final result should be such that exactly one agent responds with 'Yes' and all others say 'No' as there should be exactly one leader."
        )
        self.question_for_prediction = "Are you the leader?"
        # super().__init__(graph, model_name, model_provider, chain_of_thought)

    def get_score(self, answers: list[str]) -> float:
        valid_values = {"No", "Yes"}
        all_valid = all(x in valid_values for x in answers)
        one_leader = len([x for x in answers if x == "Yes"]) == 1
        return float(all_valid and one_leader)
    
    def get_valid_answers(self, node_id):
        return ["Yes", "No"]


class Matching(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="Qwen3-32B-FP8", model_provider="openai", chain_of_thought=True, **kwargs):
    # def __init__(self, graph, rounds, model_name="GLM-4.5-Flash", model_provider="openai", chain_of_thought=True, **kwargs):

        super().__init__(graph, rounds, model_name, model_provider, chain_of_thought, **kwargs)

        self.rounds = rounds
        self.short_task_description = "Your task is to find build groups of two agents each which can communicate with each other."
        self.long_problem_description = (
            "You will be requested to name one of your neighbors that you build a group with or 'None' if all your neighbors are already assigned to other groups and cannot be in a group with you." \
            "In the end, every agent should only be in at most one group and agents in the same group have to name each other as the second group member consistently."
        )
        self.question_for_prediction = "Please answer with the name of the neighbor you build a group with or 'None' if all your neighbors are already assigned to other groups."
        # super().__init__(graph, model_name, model_provider, chain_of_thought)

    def get_score(self, answers: list[str]) -> float:
        graph = self.graph
        node_names = [graph.nodes[node]['name'] for node in graph.nodes]
 
        name_to_match = {node_names[i]: answers[i] for i in range(len(node_names))}

        inconsistent_count = 0
        for node in graph.nodes:
            matching_node = answers[node]
            if matching_node != 'None':
                if matching_node not in [node_names[u] for u in self.neighbor_ids(node)]:
                    inconsistent_count += 1
                elif name_to_match[matching_node] != node_names[node]:
                    inconsistent_count += 1
            else:
                for v in self.neighbor_ids(node):
                    if answers[v] == 'None':
                        inconsistent_count += 1
                        break

        return (graph.order() - inconsistent_count) / graph.order()
    
    def get_valid_answers(self, node_id):
        return [self.graph.nodes[neigh]["name"] for neigh in self.neighbor_ids(node_id)] + ["None"]


class Coloring(LiteralMessagePassing):
    def __init__(self, graph: nx.Graph, rounds: int, num_colors: int | None = None, model_name="Qwen3-32B-FP8", model_provider="openai", chain_of_thought=True, **kwargs):
    # def __init__(self, graph: nx.Graph, rounds: int, num_colors: int | None = None, model_name="GLM-4.5-Flash", model_provider="openai", chain_of_thought=True, **kwargs):
    
        super().__init__(graph, rounds, model_name, model_provider, chain_of_thought, **kwargs)
        # super().__init__(graph, model_name, model_provider, chain_of_thought)
        self.rounds = rounds


        if num_colors is not None:
            self.num_colors = num_colors
        else:
            UG = graph.to_undirected() if graph.is_directed() else graph
            self.num_colors = max(UG.degree(n) for n in UG.nodes) + 1

        self.colors = [f"Group {i+1}" for i in range(self.num_colors)]

        self.short_task_description = "Your task is to partition yourselves into groups such that agents who are neighbors are never in the same group."
        self.long_problem_description = (
            f"You will be requested to state which group you assign yourself to. There are exactly {self.num_colors} groups available: Group 1,...,Group {self.num_colors}. You should assign yourself to exactly one of these groups. "
            "The final result should be such that any two agents who are neighbors are in different groups. In particular, you should assign yourself to a group that is different from all of your neighbors' groups. "
        )
        self.question_for_prediction = f"Which group do you assign yourself to?"

    def get_score(self, answers: list[str]) -> float:
        all_valid = all(x in self.colors for x in answers)
        UG = self.graph.to_undirected() if self.graph.is_directed() else self.graph
        valid_edges = [answers[u] != answers[v] for (u, v) in UG.edges].count(True)
        valid_ratio = valid_edges / UG.number_of_edges() if UG.number_of_edges() > 0 else 1.0
        return float(all_valid) * valid_ratio
    
    def get_valid_answers(self, node_id):
        return self.colors


def score_vertex_cover(results, graph):
    UG = graph.to_undirected() if graph.is_directed() else graph
    def vertex_cover(results):
        covered_edges = sum([1 for u, v in UG.edges if (results[u] == "Yes" or results[v] == "Yes")])
        return covered_edges / UG.number_of_edges() if UG.number_of_edges() > 0 else 1.0

    coverage = vertex_cover(results)

    minimality = 0
    cover_size = 0
    for u in graph.nodes:
        if results[u] is not None:
            if results[u] == "Yes":
                cover_size += 1
                _results = results.copy()
                _results[u] = "No"
                if vertex_cover(_results) < 1.0:
                    minimality += 1

    return coverage * minimality / cover_size if cover_size > 0 else 0.0


class VertexCover(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="Qwen3-32B-FP8", model_provider="openai", chain_of_thought=True, **kwargs):
    # def __init__(self, graph, rounds, model_name="GLM-4.5-Flash", model_provider="openai", chain_of_thought=True, **kwargs):    
        """https://math.stackexchange.com/a/1764484.
        A practical example is that the minimal vertex cover receives resources and is 
        important that every channel of communication always has access to this resource,
        meaning there is no need for two-hop communication to obtain some resource. Fundamentally,
        the agents solve a resource allocation problem, which also touches into fairness.
        """
        # super().__init__(graph, model_name, model_provider, chain_of_thought)
        super().__init__(graph, rounds, model_name, model_provider, chain_of_thought, **kwargs)
        self.rounds = rounds
        self.short_task_description = """Your task is to select, among all agents, a group of coordinators
such that whenever two agents communicate at least one of them is a coordinator. The group of coordinators
should be selected such that every coordinator has at least one neighbor who is not a coordinator.
"""
        # NOTE: Closer to definition
        # reverting any coordinator back to a regular agent results in at least two agents 
        # who can communicate but none of whom is a coordinator.

        self.long_problem_description = (
            "You will be requested to state whether you are a coordinator. The response should either be 'Yes' or 'No'. "
        )
        self.question_for_prediction = "Are you a coordinator?"

    def is_vertex_cover(self, results):
        UG = self.graph.to_undirected() if self.graph.is_directed() else self.graph
        covered_edges = sum([1 for u, v in UG.edges if (results[u] == "Yes" or results[v] == "Yes")])
        return covered_edges == UG.number_of_edges()
    
    def get_score(self, results: list[str]) -> float:
        return score_vertex_cover(results, self.graph)
    
    def get_valid_answers(self, node_id):
        return ["Yes", "No"]
