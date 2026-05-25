# 🌉 记忆之桥 Memory-Bridge

基于 RAG 的陪伴式 AI 系统，通过微信聊天记录重构逝者/故人的数字回音。

![demo](docs/demo.png)

## 核心功能

- **人格画像提取** — 上传微信聊天记录，自动提炼五层人格结构
- **记忆检索（RAG）** — BGE 中文向量模型 + ChromaDB，根据对话上下文检索最相关的历史片段
- **风格镜像** — 模仿原主的说话节奏、口头禅、表情习惯
- **反幻觉机制** — 禁止 AI 编造不存在的生活细节，所有回复均锚定真实聊天记录
- **伦理锁** — 检测轻生倾向时以亲人口吻进行现实拉回

## 技术栈

Streamlit / DeepSeek API / ChromaDB / BGE-base-zh-v1.5 / SQLite

## 快速部署

```bash
git clone <repo-url>
cd Memory-Bridge
pip install -r requirements.txt
# 编辑 .env 配置 DEEPSEEK_API_KEY
streamlit run app.py
```

## 免责声明

本项目仅供情感过渡陪伴，不替代专业心理咨询。如有严重心理困扰，请及时寻求专业帮助。
