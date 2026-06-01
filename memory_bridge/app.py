"""
Memory-Bridge（记忆之桥）
基于 RAG 的陪伴式 AI 系统 — 通过聊天记录重构故人的数字回音。

技术栈：Streamlit + DeepSeek API + ChromaDB + BGE Embedding
架构：上传聊天记录 → 人格提炼 → 向量化存储 → RAG 检索 → 风格镜像回复
"""

from __future__ import annotations

import html
import os
import traceback
import uuid
from datetime import datetime

import streamlit as st

from config import AVATAR_USER, AVATAR_ASSISTANT, LAST_SESSION_PATH, project_root, setup_env
from db import (
    init_db,
    load_messages_from_db,
    save_message_to_db,
    clear_chat_history_db,
    clear_memory_db,
    load_memory_from_db,
    save_memory_to_db,
)
from embedding import clear_chroma_session
from llm import (
    _friendly_llm_error_message,
    extract_persona_from_txt,
    generate_ai_response,
)
from text_processing import extract_top_names, format_wechat_time


# ---------------------------------------------------------------------------
# UI 辅助函数
# ---------------------------------------------------------------------------

def load_css() -> None:
    """从 style.css 读取全局样式并注入页面。"""
    css_path = os.path.join(project_root, "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """初始化 session_state，通过 URL Query Params + .last_session 文件持久化 Session。

    恢复优先级：
      1. URL 中已有 ?sid=xxxx → 复用
      2. .last_session 文件中记录了上次的 session_id 且 DB 中有数据 → 复用
      3. 以上都不满足 → 生成新 UUID
    确定 session_id 后写入 .last_session 文件，确保关闭浏览器后重新打开可恢复。
    """
    if "session_id" not in st.session_state:
        sid_from_url = st.query_params.get("sid")
        if sid_from_url:
            st.session_state.session_id = sid_from_url
        else:
            # 尝试从 .last_session 文件恢复上一次的 session_id
            recovered = None
            if os.path.isfile(LAST_SESSION_PATH):
                try:
                    with open(LAST_SESSION_PATH, "r", encoding="utf-8") as f:
                        candidate = f.read().strip()
                    # 验证该 session_id 在数据库中是否有数据
                    if candidate:
                        memory = load_memory_from_db(candidate)
                        messages = load_messages_from_db(candidate)
                        if memory or messages:
                            recovered = candidate
                except Exception:
                    pass
            if recovered:
                st.session_state.session_id = recovered
            else:
                st.session_state.session_id = str(uuid.uuid4())
            st.query_params["sid"] = st.session_state.session_id

    sid = st.session_state.session_id

    # 每次确定 session_id 后持久化到 .last_session 文件
    try:
        with open(LAST_SESSION_PATH, "w", encoding="utf-8") as f:
            f.write(sid)
    except Exception:
        pass

    # 恢复角色名字状态
    if "ai_name" not in st.session_state:
        st.session_state.ai_name = st.query_params.get("ai_name", "Ta")
    if "user_name" not in st.session_state:
        st.session_state.user_name = st.query_params.get("user_name", "我")

    # 刷新后恢复单选框选项
    ai_n = st.query_params.get("ai_name")
    user_n = st.query_params.get("user_name")
    if ai_n and user_n and not st.session_state.get("detected_names"):
        st.session_state.detected_names = [ai_n, user_n]

    if "memory_data" not in st.session_state:
        st.session_state.memory_data = load_memory_from_db(sid)
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None


# ---------------------------------------------------------------------------
# 弹窗组件
# ---------------------------------------------------------------------------

@st.dialog("角色选择")
def confirm_role_dialog(detected_names):
    st.write("系统检测到以下联系人，请问您想重塑谁的数字回音？")
    selected = st.radio(
        "联系人",
        detected_names,
        label_visibility="collapsed",
        key="role_radio",
    )
    if st.button("确认", type="primary"):
        # 当 detected_names 中存在重名时，过滤结果可能为空，fallback 到 "我"
        others = [n for n in detected_names if n != selected]
        other = others[0] if others else "我"
        st.session_state.ai_name = selected
        st.session_state.user_name = other
        st.query_params["ai_name"] = selected
        st.query_params["user_name"] = other
        st.session_state.pending_role_selection = None
        st.rerun()


@st.dialog("⚠️ 记忆重构确认")
def confirm_upload_dialog(uploaded_file):
    st.write(
        f"检测到新语料：`{uploaded_file.name}`。\n"
        "**是否要清空当前聊天记录，并重新注入新记忆？**"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("是 (清空并重构)", type="primary"):
            # 1. 执行清空逻辑
            clear_chat_history_db(st.session_state.session_id)

            # 2. 执行 RAG 提取与人格分析
            try:
                raw = uploaded_file.getvalue()
                decoded = raw.decode("utf-8", errors="replace")

                with st.spinner("⏳ 正在重构数字灵魂，这可能需要约1分钟..."):
                    persona = extract_persona_from_txt(
                        decoded, st.session_state.session_id
                    )
                    st.session_state.memory_data = persona
                    save_memory_to_db(st.session_state.session_id, persona)

                    top_names = extract_top_names(decoded)
                    st.session_state.detected_names = top_names

                if len(top_names) >= 2:
                    st.session_state.pending_role_selection = top_names
                    st.rerun()
                else:
                    st.toast("数字灵魂已重构！", icon="✅")
                    st.rerun()
            except Exception as exc:
                traceback.print_exc()
                st.error(_friendly_llm_error_message(exc))

    with col2:
        if st.button("否 (取消)"):
            # 取消操作，关闭弹窗并刷新页面
            st.rerun()


@st.dialog("🧹 清空历史对话")
def confirm_clear_chat_dialog():
    st.write("确定要清空所有聊天记录吗？记忆不会被清除。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确定清空", type="primary"):
            clear_chat_history_db(st.session_state.session_id)
            st.toast("历史对话已清空，页面即将刷新...", icon="✅")
            st.rerun()
    with col2:
        if st.button("取消"):
            st.rerun()


@st.dialog("🔄 完全重置")
def confirm_full_reset_dialog():
    st.write("⚠️ 此操作将永久删除所有记忆和对话记录，不可恢复。确定继续吗？")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确定重置", type="primary"):
            sid = st.session_state.session_id
            clear_chat_history_db(sid)
            clear_memory_db(sid)
            clear_chroma_session(sid)
            st.session_state.memory_data = None
            st.session_state.detected_names = []
            st.session_state.processed_file = None
            st.session_state.ai_name = "Ta"
            st.session_state.user_name = "我"
            st.query_params.pop("ai_name", None)
            st.query_params.pop("user_name", None)
            st.toast("已完全重置，页面即将刷新...", icon="✅")
            st.rerun()
    with col2:
        if st.button("取消"):
            st.rerun()


# ---------------------------------------------------------------------------
# 主要 UI 组件
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.markdown(
        '<div style="font-size: 18px; font-weight: 600; letter-spacing: 1px; color: #E5E5E5; margin-bottom: 4px;">'
        "🌉 记忆之桥</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<small style='color: #9CA3AF;'>*枯萎的文字已开出数字的花，愿它能陪你等来下一个春天。*</small>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    uploaded = st.sidebar.file_uploader(
        "注入记忆 (上传历史聊天记录)",
        type=["txt"],
        help="上传与Ta的聊天记录（微信导出的 TXT 文件），用于重塑数字回音。",
    )
    st.sidebar.caption("数据仅在本地处理。请确保您已获得相关亲属的明确授权，切勿擅自上传他人私密记忆。")
    # Step 1: 拦截新文件并唤醒屏幕居中模态弹窗
    if uploaded is not None and uploaded.name != st.session_state.get("processed_file"):
        # 标记为已处理，防止重复弹窗
        st.session_state.processed_file = uploaded.name
        # 唤醒屏幕居中模态弹窗
        confirm_upload_dialog(uploaded)

    if st.session_state.get("pending_role_selection"):
        names = st.session_state.pending_role_selection
        confirm_role_dialog(names)

    if st.session_state.memory_data:
        st.sidebar.success("✅ 当前已挂载数字记忆 (已持久化保护)")

    st.sidebar.divider()
    if st.sidebar.button("🧹 清空历史对话"):
        confirm_clear_chat_dialog()

    st.sidebar.divider()
    if st.sidebar.button("🔄 完全重置", type="primary"):
        confirm_full_reset_dialog()

    st.sidebar.divider()
    st.sidebar.caption("AI重构的回音存在偏差，仅供情感慰藉，请勿替代现实。")
    st.sidebar.caption("v0.2.0 · Memory-Bridge")


def render_chat_messages(messages: list[dict]) -> None:
    """纯渲染函数：遍历消息列表渲染聊天气泡和时间戳。"""
    if not messages:
        return

    last_time: datetime | None = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ts_str = msg.get("timestamp", "")

        # 5分钟聚合防抖：间隔 >300s 才渲染时间戳
        try:
            current_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            current_dt = None

        if current_dt is not None:
            if last_time is None or (current_dt - last_time).total_seconds() > 300:
                st.markdown(
                    f'<div style="text-align: center; color: #777777; font-size: 12px; margin: 10px 0;">{format_wechat_time(ts_str)}</div>',
                    unsafe_allow_html=True,
                )
                last_time = current_dt

        if role == "assistant":
            with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
                st.markdown(f'<div class="mb-bubble mb-ai">{html.escape(content)}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("user", avatar=AVATAR_USER):
                st.markdown(f'<div class="mb-bubble mb-user">{html.escape(content)}</div>', unsafe_allow_html=True)


def _update_header_title(title: str) -> None:
    """动态更新原生 Header 的中心文本。"""
    st.markdown(f"""
    <style>
        [data-testid="stHeader"]::after {{
            content: "{title}";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #E5E5E5;
            font-size: 16px;
            font-weight: 500;
            letter-spacing: 1px;
            pointer-events: none;
        }}
    </style>
    """, unsafe_allow_html=True)


@st.fragment
def _render_chat_area(ai_name: str) -> None:
    """聊天区域 fragment：渲染消息列表、空状态，处理流式对话。"""
    sid = st.session_state.session_id
    messages = load_messages_from_db(sid)
    render_chat_messages(messages)

    # 未上传记忆且无聊天记录时显示空状态引导页
    has_messages = len(messages) > 0
    if not st.session_state.get("memory_data") and not has_messages:
        st.markdown(
            """
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                min-height: 50vh;
                color: #6B7280;
                user-select: none;
            ">
                <div style="font-size: 64px; margin-bottom: 24px; opacity: 0.6;">🌉</div>
                <div style="font-size: 22px; font-weight: 500; color: #9CA3AF; margin-bottom: 10px; letter-spacing: 2px;">
                    将回忆化为回音
                </div>
                <div style="font-size: 14px; color: #6B7280; line-height: 1.8;">
                    点击左上角 <span style="color: #9CA3AF; font-weight: 500;">☰</span> 菜单，上传Ta的聊天记录，开始对话
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 处理由 render_main 中 st.chat_input 触发的待发送消息
    pending = st.session_state.pop("_pending_input", None)
    if pending is None:
        return

    save_message_to_db(sid, "user", pending)

    # 即时渲染用户消息
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(
            f'<div class="mb-bubble mb-user">{html.escape(pending)}</div>',
            unsafe_allow_html=True,
        )

    # 流式输出 AI 回复
    _update_header_title("对方正在输入...")

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="mb-bubble mb-ai" style="color: transparent; min-height: 24px;">...</div>',
            unsafe_allow_html=True,
        )
        full_text = ""
        for chunk in generate_ai_response(pending):
            if chunk:
                full_text += chunk
                placeholder.markdown(
                    f'<div class="mb-bubble mb-ai">{html.escape(full_text)}</div>',
                    unsafe_allow_html=True,
                )

    save_message_to_db(sid, "assistant", full_text)

    _update_header_title(ai_name)


def render_main() -> None:
    ai_name = st.session_state.get("ai_name", "Ta")
    _update_header_title(ai_name)
    _render_chat_area(ai_name)

    # chat_input 必须放在 fragment 外部，Streamlit 才能将其固定在视口底部
    prompt = st.chat_input("想对Ta说些什么...")
    if prompt is None:
        return

    chat_text = prompt.strip()
    if not chat_text:
        return

    st.session_state._pending_input = chat_text
    st.rerun()


def main() -> None:
    setup_env()
    st.set_page_config(
        page_title="记忆之桥 Memory-Bridge",
        page_icon="🌉",
        layout="centered",
    )

    init_db()
    load_css()
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
