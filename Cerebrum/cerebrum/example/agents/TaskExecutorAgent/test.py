import unittest
from cerebrum.example.agents.TaskExecutorAgent.agent import TaskExecutorAgent

class TestTaskExecutorAgent(unittest.TestCase):

    def setUp(self):
        self.agent_name = "task_executor_agent"
        self.agent = TaskExecutorAgent(self.agent_name)

    def test_execute_task(self):
        task_type = "leader_election"
        n_agents = 5
        
        result = self.agent.run(task_type, n_agents)

        self.assertIn("result", result)
        self.assertIn("rounds", result)
        self.assertTrue(result["rounds"] > 0)  # 检查轮次是否大于 0
        self.assertIsInstance(result["result"], str)  # 检查返回的结果类型

    def test_invalid_task(self):
        task_type = ""
        n_agents = 5

        result = self.agent.run(task_type, n_agents)

        self.assertIn("result", result)
        self.assertEqual(result["result"], "任务执行失败。")

if __name__ == '__main__':
    unittest.main()
