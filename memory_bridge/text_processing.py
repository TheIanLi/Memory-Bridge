"""
文本处理工具：微信记录解析、噪音清洗、时间格式化
"""

import re
from collections import Counter
from datetime import datetime

from config import LLM_MODEL_NAME, RE_DATE_PREFIX, RE_PLACEHOLDER


def extract_top_names(text: str) -> list[str]:
    """
    匹配微信记录格式中 ] 之后、(wxid) 之前的真实姓名。

    降级策略：
      1. 正则匹配 ] 姓名(wxid) 格式
      2. 正则匹配行首 姓名：/：格式
      3. 仍失败 → 调用 DeepSeek API 用 LLM 识别（产生一次额外 API 调用）
    """
    names = re.findall(r"]\s*([^\s(]+?)\(", text)
    names = [name.strip() for name in names if name.strip() and len(name) >= 2]
    if not names:
        names = re.findall(r"^([^\s:：]{2,10})[：:]", text, re.MULTILINE)
        names = [name.strip() for name in names if name.strip() and len(name) >= 2]
    if names:
        return [item[0] for item in Counter(names).most_common(2)]

    # 三级 fallback：正则完全 miss 时用 LLM 识别
    try:
        from llm import _build_deepseek_client

        client = _build_deepseek_client()
        snippet = text[:2000] if len(text) > 2000 else text
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    "以下是一段聊天记录的开头，请识别其中的两个对话参与者姓名/昵称，"
                    f"只输出两个名字用逗号分隔，不要其他内容。\n\n{snippet}"
                ),
            }],
            max_tokens=100,
            temperature=0,
            stream=False,
        )
        content = response.choices[0].message.content
        if content:
            llm_names = [n.strip() for n in content.split(",") if n.strip()]
            return llm_names[:2]
    except Exception:
        pass
    return []


def local_clean_chat_text(raw_text: str) -> str:
    """
    纯本地正则清洗微信/聊天记录中的无意义噪音，提升送入大模型的数据纯度。

    清洗项目：
    1. 剔除 [图片]、[表情]、[语音]、[视频] 等占位符
    2. 剔除冗长的日期和时间戳前缀（如 [2024-05-04 14:00:00]）
    3. 合并多余的空行
    """
    if not raw_text:
        return raw_text

    # 1. 删除无意义占位符
    cleaned = RE_PLACEHOLDER.sub("", raw_text)

    # 2. 删除行首的日期时间戳前缀
    cleaned = RE_DATE_PREFIX.sub("", cleaned)

    # 3. 合并多余空行：3 个及以上连续换行合并为 2 个换行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def format_wechat_time(dt_str: str) -> str:
    """将数据库标准 UTC/Local 时间转化为微信口语化时间"""
    if not dt_str:
        return ""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

    now = datetime.now()
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    elif dt.year == now.year:
        return dt.strftime("%m月%d日 %H:%M")
    else:
        return dt.strftime("%Y年%m月%d日 %H:%M")
