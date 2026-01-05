import os, json, asyncio, random
import networkx as nx
import logging

def determine_rounds(task, graph, default_rounds):
        UG = graph.to_undirected() if graph.is_directed() else graph
        if UG.number_of_nodes() == 0:
            return default_rounds
        if not nx.is_connected(UG):
            largest_cc = max(nx.connected_components(UG), key=len)
            UG = UG.subgraph(largest_cc).copy()
        d = nx.diameter(UG)
        if task in ["consensus", "leader_election"] or graph.number_of_nodes() > 16:
            return 2 * d + 1
        return default_rounds

class TaskExecutorAgent:

    def __init__(self, agent_name):
        self.agent_name = agent_name
        # logging.basicConfig(level=logging.DEBUG)
        # logging.debug(f"TaskExecutorAgent initialized with {self.agent_name}")

    def run(self, task_input):
        # logging.debug(f"TaskExecutorAgent running with input: {task_input}")
        """
        task_input 支持 dict 或 str(json)：
        {
          "task_type": "...",
          "topology": "chain|star|tree|net|mlp|random",
          "n_agents": 12,
          "rounds": 4,
          "seed": 42,
          "model_name": "qwen3-32b-awq",
          "model_provider": "openai",
          "chain_of_thought": true
        }
        """
        from .topo import LiteralMessagePassing as lmp
        from .topo.utils import (
            relabel_and_name_vertices,
            generate_chain_graph, generate_star_graph, generate_tree_graph,
            generate_net_graph, generate_mlp_graph, generate_random_dag,
        )
        from .topo.task_protocols import build_system_prompt as proto_build_prompt
        from .topo.task_protocols import parse_agent_final_json as proto_parse_final
        from .topo.task_protocols import aggregate as proto_aggregate
        if isinstance(task_input, str):
            task_input = json.loads(task_input)

        TASKS = {
                "matching": lmp.Matching,
                "consensus": lmp.Consensus,
                "coloring": lmp.Coloring,
                "leader_election": lmp.LeaderElection,
                "vertex_cover": lmp.VertexCover,
            }

        GEN_MAP = {
                "chain": generate_chain_graph,
                "star": generate_star_graph,
                "tree": generate_tree_graph,
                "net": generate_net_graph,
                "mlp": generate_mlp_graph,
                "random": generate_random_dag,
            }

        task_type = task_input["task_type"]
        topology = task_input["topology"]
        n_agents = int(task_input["n_agents"])
        seed = int(task_input.get("seed", 42))
        default_rounds = int(task_input.get("rounds", 4))
        model_name = task_input.get("model_name", "qwen3-32b-awq")
        model_provider = task_input.get("model_provider", "openai")
        chain_of_thought = bool(task_input.get("chain_of_thought", True))

        if task_type not in TASKS:
            return {"agent_name": self.agent_name, "result": f"未知 task_type: {task_type}"}
        if topology not in GEN_MAP:
            return {"agent_name": self.agent_name, "result": f"未知 topology: {topology}"}

        random.seed(seed)

        # 1) 造图 + 命名
        gen_fn = GEN_MAP[topology]
        try:
            graph = gen_fn(n_agents, seed=seed, directed=True)
        except TypeError:
            graph = gen_fn(n_agents)
        graph = relabel_and_name_vertices(graph)

        # 2) 轮数估计（复现 main.py）
        rounds = determine_rounds(task_type, graph, default_rounds)

        # 3) LMP 初始化：关键是 protocol hooks（task_protocols）
        task_class = TASKS[task_type]
        lmp_model = task_class(
            graph=graph,
            rounds=rounds,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            
            agent_name_prefix=self.agent_name,
            use_aios_kernel=True,

            protocol_enabled=True,
            protocol_task_name=task_type,
            protocol_build_prompt=lambda neighs, r: proto_build_prompt(task_type, neighs, r),
            protocol_parse_final=proto_parse_final,
            # 过滤 None，避免聚合器炸
            protocol_aggregate=lambda agent_jsons, G: proto_aggregate(
                task_type, [x for x in agent_jsons if isinstance(x, dict)], G
            ),
            
        )

        async def _run_async():
            await lmp_model.bootstrap()
            answers = await lmp_model.pass_messages()
            score = lmp_model.get_score(answers)
            final_answer = lmp_model.get_final_answer()
            return answers, score, final_answer

        try:
            answers, score, final_answer = asyncio.run(_run_async())
            return {
                "agent_name": self.agent_name,
                "task_type": task_type,
                "topology": topology,
                "n_agents": n_agents,
                "rounds": rounds,
                "model_name": model_name,
                "score": score,
                "final_answer": final_answer,
                "successful": True,
            }
        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "task_type": task_type,
                "topology": topology,
                "n_agents": n_agents,
                "rounds": rounds,
                "model_name": model_name,
                "successful": False,
                "error_message": f"{type(e).__name__}: {e}",
            }
