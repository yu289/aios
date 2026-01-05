# import argparse
# from agents.test_agent.agent import TestAgent
# from agents.chat_agent.agent import ChatAgent

# def main():
#     parser = argparse.ArgumentParser(description="Run test agent and chat agent together")
#     parser.add_argument("--task_input", type=str, required=True, help="Task input for the agent")
#     args = parser.parse_args()

#     # 创建并运行 TestAgent
#     test_agent = TestAgent("test_agent")
#     test_agent_result = test_agent.run(args.task_input)

#     # 创建并运行 ChatAgent，使用 TestAgent 的结果
#     chat_agent = ChatAgent("chat_agent")
#     final_answer = chat_agent.run(args.task_input, test_agent_result)

#     print("Final Answer:", final_answer)

# if __name__ == "__main__":
#     main()
import unittest
from cerebrum.example.agents.TaskClassifierAgent.agent import TaskClassifierAgent
from cerebrum.example.agents.TaskExecutorAgent.agent import TaskExecutorAgent
from cerebrum.example.agents.ResultTranslatorAgent.agent import ResultTranslatorAgent
import json

class TestAIOSIntegration(unittest.TestCase):
    
    def setUp(self):
        # 初始化
        self.task_classifier = TaskClassifierAgent("task_classifier_agent")
        self.task_executor = TaskExecutorAgent("task_executor_agent")
        self.result_translator = ResultTranslatorAgent("result_translator_agent")
    
    def run_workflow(self, task_text, n_agents):
        """
        该函数   完整工作流
        """
        # TaskClassifierAgent 任务分类
        task_input = {"task_text": task_text, "n_agents": n_agents}
        task_classification = self.task_classifier.run(json.dumps(task_input, ensure_ascii=False))
        print(f"Task classification result: {task_classification}")
        
        # 根据任务分类结果生成任务执行输入
        task_type = task_classification["task_type"]
        topology = task_classification["topology"]
        n_agents = task_classification["n_agents"]
        
        exec_input = {
            "task_type": task_type,
            "topology": topology,
            "n_agents": n_agents,
            "seed": 42,
            "rounds": 4,
            "model_name": "Qwen3-32B-FP8",
            # "model_name": "GLM-4.5-Flash",
            "model_provider": "openai",
            "chain_of_thought": True
        }
        
        # TaskExecutorAgent 执行任务
        task_execution = self.task_executor.run(json.dumps(exec_input, ensure_ascii=False))
        print(f"Task execution result: {task_execution}")
        
        # 根据执行结果生成结果翻译输入
        final_answer = task_execution["final_answer"]
        result_translation_input = {
            "task_type": task_type,
            "final_answer": final_answer,
            "meta": {
                "score": task_execution["score"],
                "rounds": task_execution["rounds"],
                "topology": task_execution["topology"]
            }
        }
        
        # ResultTranslatorAgent 翻译结果
        result_translation = self.result_translator.run(json.dumps(result_translation_input, ensure_ascii=False))
        print(f"Result translation: {result_translation}")
        
        return result_translation

    def test_full_workflow(self):
        # task_text = "请帮我选举一个领导节点"
        task_text = "在一个通信网络中，每个节点需要被分配一个信道编号，以便同时传输数据时不会产生干扰。请设计一种分配方案，使得相邻节点的信道不同。"
        n_agents = 4
        
        # 运行工作流
        result_translation = self.run_workflow(task_text, n_agents)
        
        # 断言：确保翻译内容符合预期
        # self.assertIn("result", result_translation)
        # self.assertTrue(result_translation["headline"].startswith("领导者"))  # 确保翻译内容符合预期
    
    def test_invalid_workflow(self):
        invalid_task_input = {"task_text": "", "n_agents": 0}
        
        task_classification = self.task_classifier.run(json.dumps(invalid_task_input, ensure_ascii=False))
        
        self.assertEqual(task_classification["result"], "输入必须包含 task_text 和 n_agents(>0)")

if __name__ == '__main__':
    unittest.main()
