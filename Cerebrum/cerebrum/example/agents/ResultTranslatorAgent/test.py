import unittest
from cerebrum.example.agents.ResultTranslatorAgent.agent import ResultTranslatorAgent

class TestResultTranslatorAgent(unittest.TestCase):

    def setUp(self):
        self.agent_name = "result_translator_agent"
        self.agent = ResultTranslatorAgent(self.agent_name)

    def test_translate_result(self):
        task_type = "leader_election"
        final_answer = {"leader": "node_1", "votes": {"node_1": 5, "node_2": 3}}

        result = self.agent.run(task_type, final_answer)

        self.assertIn("result", result)
        self.assertIsInstance(result["result"], str)
        self.assertTrue(result["result"].startswith("领导者:"))  # 检查翻译结果的内容

    def test_invalid_input(self):
        task_type = ""
        final_answer = {}

        result = self.agent.run(task_type, final_answer)

        self.assertIn("result", result)
        self.assertEqual(result["result"], "翻译失败，输入结果无效。")

if __name__ == '__main__':
    unittest.main()
