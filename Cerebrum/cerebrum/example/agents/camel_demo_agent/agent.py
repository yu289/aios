# cerebrum/example/agents/camel_demo_agent/agent.py

from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func

from cerebrum.utils.communication import send_request

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelType, RoleType

from cerebrum.community.adapter.camel_adapter import AiosCamelModelBackend


class CamelAgent:

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        # 告诉 Cerebrum 当前用 CAMEL 框架
        prepare_framework(FrameworkType.Camel)

    def run(self, task: str):
        # 把 AIOS 的 request 函数注册进去
        set_request_func(send_request, self.agent_name)

        teacher_model = AiosCamelModelBackend(
            agent_name=self.agent_name,
            model_type=ModelType.GPT_4O_MINI,
        )
        student_model = AiosCamelModelBackend(
            agent_name=self.agent_name,
            model_type=ModelType.GPT_4O_MINI,
        )

        teacher_system_message = BaseMessage(
            role_name="Cathy",
            role_type=RoleType.ASSISTANT,
            content="You are Cathy, a teacher. You teach the student how to solve problems.",
            meta_dict={"temperature": 0.7},
        )

        student_system_message = BaseMessage(
            role_name="Joe",
            role_type=RoleType.USER,
            content="You are Joe, a student.",
            meta_dict={"temperature": 0.7},
        )

        teacher = ChatAgent(
            system_message=teacher_system_message,
            model=teacher_model,
        )

        student = ChatAgent(
            system_message=student_system_message,
            model=student_model,
        )

        # 这个 message 是“外部真实用户”对 student 说的话，所以用 USER 就行
        user_msg = BaseMessage(
            role_name="User",
            role_type=RoleType.USER,
            content=task,
            meta_dict={"temperature": 0.7},
        )

        # 角色扮演一两轮
        student_reply = student.step(user_msg)          # 学生先根据任务说点什么
        teacher_reply = teacher.step(student_reply.msg) # 老师根据学生的话进行讲解
        student_follow_up = student.step(teacher_reply.msg)  # 学生再追问或总结

        # return {
        #     "first_student_msg": student_reply.msg.to_openai_message(),
        #     "teacher_msg": teacher_reply.msg.to_openai_message(),
        #     "final_student_msg": student_follow_up.msg.to_openai_message(),
        # }
        return {
            "first_student_msg": student_reply.msg.content,
            "teacher_msg": teacher_reply.msg.content,
            "final_student_msg": student_follow_up.msg.content,
        }

