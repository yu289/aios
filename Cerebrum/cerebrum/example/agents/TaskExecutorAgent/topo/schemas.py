from typing import Dict, Any

# 定义每个任务：agent 最终一轮输出的 JSON 结构 & 系统聚合后的业务答案结构
TASK_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "coloring": {
        "agent_final_schema": {
            "type": "object",
            "required": ["node", "channel"],
            "properties": {
                "node": {"type": "string"},
                "channel": {"type": "integer", "minimum": 1}
            },
            "additionalProperties": False
        },
        "final_answer_schema": {
            "type": "object",
            "required": ["channels", "valid"],
            "properties": {
                "channels": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 1}
                },
                "valid": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "final_answer_key": "channels"
    },

    "consensus": {
        "agent_final_schema": {
            "type": "object",
            "required": ["node", "decided"],
            "properties": {
                "node": {"type": "string"},
                "decided": {}
            },
            "additionalProperties": False
        },
        "final_answer_schema": {
            "type": "object",
            "required": ["value", "agreement"],
            "properties": {
                "value": {},
                "agreement": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "final_answer_key": "value"
    },

    "leader_election": {
        "agent_final_schema": {
            "type": "object",
            "required": ["node", "leader"],
            "properties": {
                "node": {"type": "string"},
                "leader": {"type": "string"}
            },
            "additionalProperties": False
        },
        "final_answer_schema": {
            "type": "object",
            "required": ["leader", "votes"],
            "properties": {
                "leader": {"type": "string"},
                "votes": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}}
            },
            "additionalProperties": False
        },
        "final_answer_key": "leader"
    },

    "matching": {
        "agent_final_schema": {
            "type": "object",
            "required": ["node", "partner"],
            "properties": {
                "node": {"type": "string"},
                "partner": {"type": ["string", "null"]}
            },
            "additionalProperties": False
        },
        "final_answer_schema": {
            "type": "object",
            "required": ["pairs", "unmatched"],
            "properties": {
                "pairs": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}},
                "unmatched": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "final_answer_key": "pairs"
    },

    "vertex_cover": {
        "agent_final_schema": {
            "type": "object",
            "required": ["node", "in_cover"],
            "properties": {
                "node": {"type": "string"},
                "in_cover": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "final_answer_schema": {
            "type": "object",
            "required": ["cover", "is_valid"],
            "properties": {
                "cover": {"type": "array", "items": {"type": "string"}},
                "is_valid": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "final_answer_key": "cover"
    },
}
