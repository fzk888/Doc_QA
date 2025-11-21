# 文档问答系统（Doc_QA）

这是一个基于检索增强生成（RAG）的文档问答服务。支持多格式文档解析（含图片 OCR）、混合检索（BGE 向量 + BM25）、重排序（BGE-reranker），并通过大语言模型生成答案（SSE 流式或一次性返回）。

## 项目功能

- 多格式文档：PDF、Word（.docx）、Markdown、TXT、PPTX、HTML、Excel、CSV、图片 OCR
- 知识库管理：创建/删除/选择知识库，独立存储与索引
- 混合检索：BGE 向量检索 + BM25 关键词检索，BGE-reranker 重排
- 智能问答：支持多轮对话、显示来源链接、推理过程、SSE 流式输出
- 引导性问题：从知识库自动生成引导提问

## 目录结构

```
Doc_QA/
├── app.py                         # 后端入口（FastAPI）
├── functions.py                   # 检索、重排、问答核心逻辑
├── Knowledge_based_async.py       # 知识库管理与向量库构建
├── bm25_search.py                 # BM25 搜索
├── document_reranker.py           # 文档重排序
├── documen_processing.py          # 文档解析（含 DOCX 图片 OCR 路径）
├── config.yaml                    # 配置文件
├── server/
│   └── index.html                 # 可选静态前端
├── add/
│   ├── morefile/                  # 其他格式解析（PPTX/HTML/Excel/图片等）
│   └── ocr/ocr_app.py             # OCR 子服务（PaddleOCR）
├── Knowledge_based/               # 知识库根（faiss_index/markdown_directory/uploads 等）
└── req.txt                        # 依赖
```

## 安装与环境

- Python 3.10+
- Windows 10/11（已验证）或 Linux/macOS
- GPU 可选（推荐，Torch + CUDA）

安装依赖：

```bash
pip install -r req.txt
```

常见依赖：`langchain`、`faiss-gpu/faiss-cpu`、`transformers`、`torch`、`fastapi`、`uvicorn`、`python-multipart`、`sentence-transformers`、`FlagEmbedding`、`jieba`、`pymupdf`、`python-docx`、`openpyxl`、`python-pptx`、`beautifulsoup4`、`pillow`、`paddleocr`。

注意事项：
- Windows 下 `datrie` 可能需要 Microsoft Visual C++ 14.0+；如不需要可移除。
- 路径建议使用绝对路径，避免分隔符差异导致问题。
- 不要在仓库中提交任何密钥；`config.yaml` 的密钥仅用于本地开发。

## 配置

编辑 `config.yaml`：

```yaml
paths:
  kb_dir: "./Knowledge_based"
  model_dir: "./model/bge-large-zh-v1.5"
  reranker_model_dir: "./model/bge-reranker-large/quietnight/bge-reranker-large"
  openai_api_base: "http://your-llm-endpoint/v1"
  openai_api_keys: "<YOUR_API_KEY>"
  ocr_service_url: "http://127.0.0.1:8001/detection_pic"
  pix2text_url: "http://127.0.0.1:8503/pix2text"

models:
  embeddings_model: "bge-large-zh-v1.5"
  reranker_model: "bge-reranker-large"
  llm_model: "deepseek-chat"

settings:
  batch_size: 1024
  device: "cpu"
  normalize_embeddings: true
  use_fp16: true
  only_chatKBQA_default: true
  temperature_default: 0.7
  enable_ocr_images: true
  enable_pdf_pix2text: true
  pic_ocr_provider: "paddle"

system:
  max_workers: 4
```

## 启动

- 后端服务（端口 `7861`）：

```bash
python app.py
```

- 可选 OCR 子服务（端口 `8001`）：

```bash
python -m uvicorn add.ocr.ocr_app:app --host 127.0.0.1 --port 8001
```

## API 使用

- 列举知识库：`GET /list_kb`
- 删除知识库：`POST /delete_kb`（`kb_name`）
- 更新向量库（上传文件）：`POST /update_vectordb`（表单：`kb_name`、`files[]`）
- 生成引导性问题：`POST /view_guiding_questions`（JSON：`kb_name`）
- 删除文件：`POST /remove_file`（表单：`kb_name`、`file_name`）
- 多文档问答：`POST /mulitdoc_qa`（JSON，支持 SSE）
- 查看日志：`GET /logs?lines=200`

示例（问答，SSE）：

```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "user", "content": "你的问题"}
  ],
  "temperature": 0.5,
  "stream": true,
  "only_chatKBQA": true,
  "kb_name": "your_kb",
  "show_source": true,
  "derivation": false,
  "multiple_dialogue": true
}
```

## 多轮对话与日志

- 多轮对话：启用 `multiple_dialogue` 后，系统会基于 `messages` 构建上下文，并在生成阶段参考历史。
- 请求追踪：服务为每个请求注入 `req_id`，日志统一使用前缀 `[req:xxxx]`。
- 结构化日志（检索/重排）：
  - 检索数量：`retrieval bge=4 bm25=8`
  - 去重后候选：`unique_docs=7`
  - 进入重排的候选：`candidates=[A.docx,B.pdf,...]`
  - 重排结果及得分：`rerank_top=[A.docx,C.md,B.pdf] scores=[0.86,0.71,0.65]`
  - 最终选中文档：`selected_docs=[...] scores=[...]`

## 文档解析与 OCR

- PDF：抽取表格/正文/图片；可启用 `enable_pdf_pix2text` 使用图片转文字兜底。
- DOCX：优先 Pandoc 转 Markdown；若图片未被解析，回退解压 `.docx` 的 `word/media/*` 并调用 `ocr_service_url` 完成 OCR；识别结果并入文本。
- Markdown/TXT/PPTX/HTML/Excel/CSV/图片：各自处理模块位于 `add/morefile/`。

## 检索与重排

- 并行执行 BGE 向量检索与 BM25 检索，按 `file_path` 去重，限制候选数量，使用 `DocumentReranker` 重排并选出 Top-N；支持显示来源链接与非流式聚合输出。

## 注意事项

- SSE 需确保反向代理禁用缓冲（响应头包含 `X-Accel-Buffering: no`）。
- 大文档解析与向量化可能耗时，请关注 `/logs` 输出。
- 生产环境请使用安全的密钥管理与访问控制，不要将密钥写入仓库。

## 许可证

[MIT License](LICENSE)