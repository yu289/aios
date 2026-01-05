import unittest
from cerebrum.example.agents.TaskClassifierAgent.agent import TaskClassifierAgent

class TestTaskClassifierAgent(unittest.TestCase):

    def setUp(self):
        self.agent_name = "task_classifier_agent"
        self.agent = TaskClassifierAgent(self.agent_name)

    def test_classify_task(self):
        # 模拟任务描述输入
        task_input = "请帮我选择一个领导节点"
        
        # 运行任务分类
        result = self.agent.run(task_input)

        # 检查返回的 JSON 结构是否正确
        self.assertIn("task_type", result)
        self.assertIn("reason", result)
        self.assertEqual(result["task_type"], "leader_election")  # 预计任务类型
        self.assertEqual(result["reason"], "领导节点选举")  # 预计原因
    
    def test_invalid_input(self):
        # 模拟无效的任务描述
        task_input = ""
        
        result = self.agent.run(task_input)
        
        # 检查如果输入无效，是否返回适当的错误或默认值
        self.assertEqual(result["result"], "无法生成有效的任务分类。")

if __name__ == '__main__':
    unittest.main()
