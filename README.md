# Doc_QA: 高性能模块化 RAG 知识库问答系统

Doc_QA 是一款基于大规模语言模型 (LLM) 的增强检索生成 (RAG) 系统。本项目经过深度重构，采用模块化设计，支持多种文档格式的高效解析、流式问答以及精准的重排序检索。

## 🌟 核心特性

- **多模态解析支持**：
    - **文档**：PDF (支持表格提取)、Word (Pandoc/python-docx 引擎)、Markdown (支持 QA 格式)。
    - **表格与图片**：Excel/CSV (集成 LlamaIndex 方案)、JPG/PNG (PaddleOCR 智能识别)。
    - **演示文稿**：PPT/PPTX (Aspose.Slides 混合解析)。
- **高性能检索架构**：
    - **混合检索**：结合 FAISS 向量检索与 BM25 关键词检索。
    - **精排优化**：集成 BGE-Reranker 模型，对检索结果进行二次相关性修正。
- **现代化技术栈**：
    - **后端**：FastAPI 提供全量异步接口，支持 StreamingResponse 流式输出。
    - **OCR 服务**：独立 PaddleOCR 服务，支持多机分布式调用。
    - **前端**：简洁直观的 Web 预览界面。

## 📁 目录结构

```text
Doc_QA/
├── core/               # 核心引擎模块
│   ├── engine.py       # RAG 检索与 QA 逻辑 (原 functions.py)
│   ├── kb_manager.py   # 知识库管理与更新 (原 Knowledge_based_async.py)
│   ├── reranker.py     # 文档重排序逻辑
│   └── search_bm25.py  # BM25 检索实现
├── parsers/            # 文档解析模块
│   ├── main_parser.py  # 主解析调度器 (支持 PDF/MD/Word/TXT)
│   ├── excel_parser.py # 基于 LlamaIndex 的 Excel/CSV 解析
│   ├── ppt_parser.py   # PPT 解析引擎
│   └── engines/        # 深度学习解析组件 (DeepDoc/RAGFlow)
├── services/           # 外部服务
│   └── ocr/            # PaddleOCR 独立服务
├── web/                # Web 前端界面
│   └── index.html
├── app.py              # FastAPI 应用程序入口
├── config.yaml         # 全局配置
├── .env                # 环境变量
└── requirements.txt    # 项目依赖
```

## 💎 核心技术亮点：高级 Excel/CSV 财务解析

本项目针对**真实跨境电商财务数据**进行了深度优化，解决了传统解析器在处理复杂 Excel 表格时的痛点：

### 1. 复杂表头映射 (Deep Header Mapping)
- **智能表头识别**：利用 `LlamaParse` 配合 `MarkdownElementNodeParser` 自动识别多级嵌套表头、跨列分布头以及合并单元格数据。
- **行列语义关联**：通过集成 DeepSeek LLM，解析器能够理解财务报表中的类项归属关系，确保在 RAG 检索时，离岸账户与对应金额、科目与子科目之间的逻辑链路不会丢失。

### 2. 多维度混合处理
- **LLM 语义切割**：不同于简单的按行粗暴切割，系统会利用 LLM 对表格进行语义分析，将每一组具备独立业务含义的数据块定义为单独的 Node，显著提升了检索的 Top-K 召回精度。
- **OCR 增益辅助**：集成自研 OCR 服务，针对 Excel 中内嵌的各类财务单据图片进行自动识别并在后台建立关联索引。

### 3. CSV 优化
- 针对长文本 CSV 文档，实现了高精度的长度控制机制，防止 Token 溢出同时保留完整的上下文语义。

## 🚀 快速开始

### 1. 环境准备
推荐使用 Conda 创建独立环境：
```bash
conda create -n nlp python=3.11
conda activate nlp
pip install -r requirements.txt
```

### 2. 配置与启动
1. **配置文件**：编辑 `config.yaml` 填写模型路径、知识库存储目录。
2. **环境变量**：在 `.env` 中填写您的 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`。
3. **启动 OCR 服务**：
   ```bash
   cd services/ocr
   python app.py
   ```
4. **启动主应用**：
   ```bash
   python app.py
   ```

## 🛠️ 模型下载
请确保已下载以下模型并配置路径：
- **Embeddings**: `bge-large-zh-v1.5`
- **Reranker**: `bge-reranker-v2-m3`

## 📄 许可证
本项目遵循 [Apache-2.0 License](LICENSE) 协议。