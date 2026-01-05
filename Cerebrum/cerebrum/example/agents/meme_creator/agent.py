import os
from cerebrum.llm.apis import llm_chat
from cerebrum.interface import AutoTool

from cerebrum.config.config_manager import config

aios_kernel_url = config.get_kernel_url()


class MemeCreator:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        # 加载你自己的本地 TextToImage 工具
        self.image_tool = AutoTool.from_preloaded(
            "text_to_image", local=True
        )

        os.makedirs("./memes", exist_ok=True)

    def run(self, task_input, config_=None):
        # 1. 让 LLM 把用户输入变成画图 prompt
        messages = [
            {
                "role": "system",
                "content": "You are a meme prompt generator. Convert user input into a short, vivid English image prompt."
            },
            {"role": "user", "content": task_input},
        ]
        resp = llm_chat(
            agent_name=self.agent_name,
            messages=messages,
            base_url=aios_kernel_url,
        )
        prompt = resp["response"]["response_message"]

        # 2. 真正调用工具生成图片
        output_path = "./memes/cat_meme.png"
        tool_result = self.image_tool.run({
            "prompt": prompt,
            "path": output_path,
        })

        # 3. 返回真实存在的路径
        return {
            "agent_name": self.agent_name,
            "result": f"Prompt: {prompt}\n{tool_result}",
            "rounds": 1,
        }
