import json
import re
import logging
from .bert_service import bert_predict
from appbuilder.core.console.appbuilder_client import AppBuilderClient
from config import APPBUILDER_TOKEN, APP_ID
from datetime import datetime
from services.statistics import rumor_detection_lock, rumor_detection_count, rumor_detection_history, MAX_HISTORY_DAYS

client = AppBuilderClient(APP_ID, secret_key=APPBUILDER_TOKEN)

def safe_json_load(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r"```[a-zA-Z]*", "", s).strip(" \n`")
        match = re.search(r"\{.*\}", s, re.S)
        if match:
            return json.loads(match.group(0))
        raise

def validate_response(data: dict):
    required_keys = {"answer", "sources_count", "top_match_rumor"}
    if not isinstance(data, dict):
        return False, "返回结果不是字典格式"
    if not required_keys.issubset(data.keys()):
        return False, f"缺少必需字段，必须包含{required_keys}"
    if not isinstance(data["sources_count"], int) or data["sources_count"] < 0:
        return False, "'sources_count' 必须是非负整数"
    if data["sources_count"] == 0:
        if not isinstance(data["answer"], str):
            return False, "sources_count为0时，answer应为字符串"
        if data["top_match_rumor"] != "":
            return False, "sources_count为0时，top_match_rumor必须是空字符串"
    else:
        if not data["answer"]:
            return False, "sources_count大于0时，answer不能为空"
        if not data["top_match_rumor"]:
            return False, "sources_count大于0时，top_match_rumor不能为空"
    return True, "校验通过"

def get_rumor_status_and_refutation(text: str):
    global rumor_detection_count, rumor_detection_history

    bert_out = bert_predict(text)
    bert_label = "谣言" if bert_out["label"] == 1 else "非谣言"
    bert_prob = round(bert_out["prob"], 3)

    prompt = (
    "你是辟谣专家，请根据以下规则和 JSON 格式输出：\n"
    "规则：\n"
    "1. 如果 sources_count 为 0，answer 可以是默认的提示信息，也可以为空，top_match_rumor 可以为空字符串 \"\"。\n"
    "2. 如果 sources_count 大于 0，answer 应给出针对谣言的辟谣说明，top_match_rumor 应填写最相关的网页谣言内容。\n"
    "3. 输出必须是 JSON 格式，不能有其他解释文字。\n"
    "4. 若用户文本较长，务必确保文本传输完整，不可丢失内容。\n"
    "示范输出格式：\n"
    '{"answer":"辟谣说明内容", "sources_count":1, "top_match_rumor":"最相关谣言内容"}\n'
    "请结合本地 BERT 模型判断：{bert_label}，置信度 {bert_prob}，\n"
    f"和用户文本（文本较长的情况下，确保完整处理）：\n"
    f"{text}\n"
    "给出综合判断结果。"
    )


    conv_id = client.create_conversation()
    if not conv_id:
        logging.error("创建对话失败")
        return {
            "is_rumor": False,
            "refutation": "对话创建失败，无法获取辟谣信息。",
            "sources_count": 0,
            "top_match_rumor": "",
        }

    resp = client.run(conv_id, prompt, stream=False)
    if not resp or not hasattr(resp, 'content') or not hasattr(resp.content, 'answer'):
        logging.error("模型返回数据异常")
        return {
            "is_rumor": False,
            "refutation": "模型返回异常，无法提供辟谣信息。",
            "sources_count": 0,
            "top_match_rumor": "",
        }

    try:
        data = safe_json_load(resp.content.answer)
    except Exception as e:
        logging.error(f"JSON解析失败: {str(e)}")
        return {
            "is_rumor": False,
            "refutation": "无法解析模型返回结果。",
            "sources_count": 0,
            "top_match_rumor": "",
        }

    # 后处理：保证格式正确
    sources_count = data.get("sources_count", 0)
    top_match_rumor = data.get("top_match_rumor", "")
    if sources_count > 0 and not top_match_rumor:
        data["top_match_rumor"] = "最相关谣言内容未提供"
    if sources_count == 0:
        data["top_match_rumor"] = top_match_rumor or ""

    valid, msg = validate_response(data)
    if not valid:
        logging.error(f"模型返回格式校验失败: {msg}")
        return {
            "is_rumor": False,
            "refutation": f"模型返回结果格式错误: {msg}",
            "sources_count": 0,
            "top_match_rumor": "",
        }

    # 模糊判据：只要bert概率大于0.5或者有来源就判为谣言
    is_rumor = (round(bert_out["prob"], 3) > 0.5) or (sources_count > 0)

    # 更新统计
    with rumor_detection_lock:
        rumor_detection_count += 1
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        if rumor_detection_history and rumor_detection_history[-1]["date"] == today_str:
            rumor_detection_history[-1]["count"] += 1
        else:
            rumor_detection_history.append({"date": today_str, "count": 1})

        if len(rumor_detection_history) > MAX_HISTORY_DAYS:
            rumor_detection_history.pop(0)

    return {
        "is_rumor": is_rumor,
        "refutation": data.get("answer", ""),
        "sources_count": sources_count,
        "top_match_rumor": data.get("top_match_rumor", ""),
    }
