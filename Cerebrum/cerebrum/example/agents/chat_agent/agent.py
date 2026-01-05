# from cerebrum.llm.apis import llm_chat
# from cerebrum.config.config_manager import config
# import argparse
# from cerebrum.interface import AutoTool


# aios_kernel_url = config.get_kernel_url()

# class ChatAgent:
#     def __init__(self, agent_name):
#         self.agent_name = agent_name
#         self.messages = []
#         self.search_tool = AutoTool.from_preloaded("wikipedia", local=True)
    
#     def _need_search(self, text: str) -> bool:
#         """简单判断是否需要联网搜索，可以根据自己需求改规则"""
#         text_lower = text.lower()
#         return (
#             "搜索" in text
#             or "查一下" in text
#             or "查查" in text
#             or "search" in text_lower
#         )

#     def run(self, task_input):
#         # 默认先认为不用搜
#         user_message = task_input

#         if self._need_search(task_input):
#             # 1. 先用工具搜索
#             query = task_input.replace("搜索", "").replace("查一下", "").strip()
#             if not query:
#                 query = task_input

#             search_result = self.search_tool.run({"query": query})

#             # 2. 把搜索结果拼到 prompt 里，让大模型总结
#             user_message = (
#                 f"用户的问题是：{task_input}\n\n"
#                 f"下面是来自 google_search 工具的搜索结果：\n{search_result}\n\n"
#                 "请你结合这些搜索结果，用中文给出清晰、可靠的回答。"
#             )
#         self.messages.append({"role": "user", "content": user_message})

#         response = llm_chat(
#             agent_name=self.agent_name,
#             messages=self.messages,
#             base_url=aios_kernel_url,
#         )

#         final_answer = response.get("response", {}).get("response_message", "")

#         self.messages.append({"role": "assistant", "content": final_answer})

#         return final_answer

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str, required=True)
#     args = parser.parse_args()

#     agent = ChatAgent("chat_agent")
#     out = agent.run(args.task)
#     print("Result:", out)
from cerebrum.llm.apis import llm_chat
from cerebrum.interface import AutoTool
from cerebrum.config.config_manager import config
import argparse

aios_kernel_url = config.get_kernel_url()

class ChatAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.messages = []
        self.search_tool = AutoTool.from_preloaded("wikipedia", local=True)
    
    def _need_search(self, text: str) -> bool:
        """判断是否需要进行搜索"""
        text_lower = text.lower()
        return "搜索" in text or "查一下" in text or "search" in text_lower

    def run(self, task_input, test_agent_result):
        """将 TestAgent 的结果和用户输入结合"""
        user_message = task_input

        if self._need_search(task_input):
            query = task_input.replace("搜索", "").replace("查一下", "").strip()
            if not query:
                query = task_input

            search_result = self.search_tool.run({"query": query})

            # 拼接搜索结果到 user_message
            user_message = (
                f"用户的问题是：{task_input}\n\n"
                f"下面是来自搜索工具的结果：\n{search_result}\n\n"
                "请结合这些信息，给出一个清晰的回答。"
            )
        
        self.messages.append({"role": "user", "content": user_message})

        # 调用 LLM 获取最终回答
        response = llm_chat(
            agent_name=self.agent_name,
            messages=self.messages,
            base_url=aios_kernel_url,
        )

        final_answer = response.get("response", {}).get("response_message", "")
        self.messages.append({"role": "assistant", "content": final_answer})

        return final_answer
