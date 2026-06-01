"""
Memory-Bridge 全局配置常量
"""

import os
import re

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 环境变量设置
# ---------------------------------------------------------------------------
_env_initialized = False


def setup_env() -> None:
    """初始化环境变量，幂等（多次调用只执行一次）。"""
    global _env_initialized
    if _env_initialized:
        return
    # 设定 HuggingFace 镜像节点，防止首次加载 BGE 模型时连接超时
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 清除 ALL_PROXY 防止 httpcore 自动拾取 SOCKS 代理导致 socksio 缺失报错
    # （项目仅通过 HTTP_PROXY 走 HTTP 代理连接 DeepSeek API）
    for _k in ("ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(_k, None)
    _env_initialized = True

# ---------------------------------------------------------------------------
# 项目根目录 & .env 加载 (防 WSL 终端路径漂移)
# ---------------------------------------------------------------------------
_config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_config_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(project_root, "memory_bridge.db")
LAST_SESSION_PATH = os.path.join(project_root, ".last_session")

# ---------------------------------------------------------------------------
# 聊天角色与极简 emoji 头像
# ---------------------------------------------------------------------------
AVATAR_USER = "👤"
AVATAR_ASSISTANT = "🌌"

# ---------------------------------------------------------------------------
# API / 对话
# ---------------------------------------------------------------------------
# 送入 API 的历史消息条数上限（user/assistant 各算一条；20 ≈ 最近 10 轮来回）
MAX_HISTORY_TURNS = 20

# DeepSeek 模型名称
LLM_MODEL_NAME = "deepseek-v4-flash"

# ---------------------------------------------------------------------------
# ChromaDB / RAG 参数
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500          # 文本切块大小（字符数）
CHUNK_OVERLAP = 50        # 相邻 chunk 重叠字符数，防止语意断层
RAG_TOP_K = 3             # 检索时返回的相关记忆片段数
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------
# 人格提炼时原始文本截取上限（清洗前），清洗后最终送入 API 的上限
MAX_CLEAN_CHARS = 100000
MAX_PERSONA_RAW_CHARS = 20000

# 日期/时间前缀正则（用于清洗阶段剔除行首时间戳）
RE_DATE_PREFIX = re.compile(
    r"\[?\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?\s+\d{1,2}:\d{2}(?::\d{2})?\]?\s*"
)
# 无意义占位符正则（[图片]、[表情]、[语音]、[视频] 等）
RE_PLACEHOLDER = re.compile(
    r"\[(?:图片|表情|语音|视频(?:\s*通话)?|[Qq]表情|[动画]表情|文件|链接|小程序|红包|转账|位置|名片|聊天记录|笔记|接龙)\]\s*"
)
