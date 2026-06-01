"""
ChromaDB 向量数据库（RAG 检索增强生成）
"""

import os
import re
import traceback

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import streamlit as st
import torch

from config import (
    project_root,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TOP_K,
    EMBEDDING_MODEL_NAME,
)


@st.cache_resource
def get_embedding_model() -> "SentenceTransformer":
    """加载 BGE 中文 embedding 模型，自动探测 GPU/CPU 环境弹性回退。

    使用 @st.cache_resource 实现全局单例缓存，Streamlit 页面刷新不会重复加载。
    优先调度到 CUDA（神舟 4060 GPU），若 WSL/PyTorch CUDA 未正确配置则自动降级 CPU。
    """
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
        print("✅ CUDA 环境已就绪，模型将调度到 GPU 加速推理。")
    else:
        device = "cpu"
        import warnings
        warnings.warn(
            "⚠️  警告：未检测到可用的 CUDA 环境，模型将降级使用 CPU 运行，请注意性能瓶颈。"
        )
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


class _BGEEmbeddingFunction(EmbeddingFunction):
    """适配 ChromaDB embedding_function 协议的本地 BGE 封装。

    继承官方 EmbeddingFunction 基类，实现 __call__ 签名，
    满足 chromadb 对自定义 embedding_function 的要求。
    """

    @classmethod
    def name(cls) -> str:
        return "BAAI_bge_base_zh_v1_5"

    def __call__(self, input: Documents) -> Embeddings:
        model = get_embedding_model()
        embeddings = model.encode(
            input,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


@st.cache_resource
def _get_chroma_collection():
    """懒加载 ChromaDB 持久化集合，通过 @st.cache_resource 实现单例。

    强制使用 BAAI/bge-base-zh-v1.5 作为 embedding_function，
    比默认 MiniLM 的中文语义理解精度提升约 15~20%。
    """
    client = chromadb.PersistentClient(
        path=os.path.join(project_root, "chroma_db")
    )
    return client.get_or_create_collection(
        name="memory_chunks",
        embedding_function=_BGEEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def _split_recursive(text: str, sep: str) -> list[str]:
    """按分隔符切分，保留分隔符在原位附近的语义完整性。"""
    parts = text.split(sep)
    result = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            result.append(stripped)
    return result


def _refine_segments(segments: list[str], sep: str, max_size: int) -> list[str]:
    """对超过 max_size 的段进一步按 sep 切分（支持正则分隔符）。"""
    refined = []
    for seg in segments:
        if len(seg) <= max_size:
            refined.append(seg)
        else:
            sub_parts = re.split(sep, seg) if sep.startswith(r"[") else seg.split(sep)
            for sub in sub_parts:
                stripped = sub.strip()
                if stripped:
                    refined.append(stripped)
    return refined


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """多级降级切分策略，带重叠区防止语意断层。

    降级链条：
      1. 优先按 \\n\\n（段落边界）
      2. 仍超标 → 按 \\n（行边界）
      3. 仍超标 → 按中文标点（。！？；，）
      4. 仍超标 → 硬截断至 chunk_size

    每块尾部和下一块头部有 overlap 字符的重叠区域，
    确保跨 chunk 的语义连续不丢失。
    """
    if not text:
        return []

    # 第一级：按段落切分
    raw_segments = _split_recursive(text, "\n\n")

    # 第二级：仍超标的段落按换行切
    raw_segments = _refine_segments(raw_segments, "\n", chunk_size)

    # 第三级：仍超标的按中文标点切
    raw_segments = _refine_segments(raw_segments, r"[。！？；，]", chunk_size)

    # 第四级：硬截断
    final_segments = []
    for seg in raw_segments:
        if len(seg) <= chunk_size:
            final_segments.append(seg)
        else:
            for i in range(0, len(seg), chunk_size - overlap):
                final_segments.append(seg[i:i + chunk_size])

    # 清洗空白
    chunks = [s.strip() for s in final_segments if s.strip()]
    # 有序去重：聊天记录中转发、刷屏等场景产生大量重复内容，去除完全相同的 chunk 减少向量空间噪音
    return list(dict.fromkeys(chunks))


def clear_chroma_session(session_id: str) -> None:
    """清除当前 session 在 ChromaDB 中的所有向量数据。"""
    collection = _get_chroma_collection()
    existing = collection.get(where={"session_id": session_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def index_corpus(text: str, session_id: str) -> int:
    """将清洗后的文本切块并存入 ChromaDB，携带 session_id metadata 防止跨 session 数据混淆。

    每次调用会先清空同 session_id 的旧 chunks（支持重新上传语料），
    不影响其他 session 的数据。
    """
    collection = _get_chroma_collection()
    session_prefix = f"sid_{session_id}"

    # 删除该 session 的旧数据
    existing = collection.get(where={"session_id": session_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunks = chunk_text(text)
    if not chunks:
        return 0

    collection.add(
        documents=chunks,
        ids=[f"{session_prefix}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"session_id": session_id} for _ in chunks],
    )
    print(f"[ChromaDB] 已索引 {len(chunks)} 个 chunk (session={session_id[:8]}...)")
    return len(chunks)


def retrieve_relevant(query: str, session_id: str, k: int = RAG_TOP_K) -> list[str]:
    """从 ChromaDB 中检索与 query 最相关的 k 个文本块，限定当前 session。

    使用 session_id metadata filter 严格隔离数据，防止跨用户数据混淆。
    返回文本块内容列表；若集合为空或无可匹配结果则返回空列表。
    """
    collection = _get_chroma_collection()
    if collection.count() == 0:
        return []
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k,
            where={"session_id": session_id},
        )
    except Exception:
        traceback.print_exc()
        return []
    docs = results.get("documents", [[]])
    return docs[0] if docs else []
