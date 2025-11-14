<<<<<<< HEAD
# 文档问答系统（Doc_QA）

这是一个基于检索增强生成（RAG）技术的文档问答系统，支持多种文档格式的上传、解析和智能问答。系统结合向量检索和关键词检索，提供高效准确的文档信息检索和智能回答功能。

## 项目功能特点

### 1. 多格式文档支持
- 支持PDF、Word（.docx/.doc）、Markdown、TXT、PPTX、HTML、Excel、CSV等多种文档格式
- 支持图片（JPG、PNG）格式的OCR文字识别

### 2. 知识库管理
- 创建、删除和选择不同的知识库
- 每个知识库独立存储和管理文档

### 3. 混合检索策略
- 结合BGE向量检索和BM25关键词检索
- 使用BGE重排序模型优化检索结果

### 4. 智能问答功能
- 基于知识库内容进行上下文感知的回答
- 支持多轮对话和历史记录
- 流式输出回答结果
- 可选择显示回答来源和推理过程

### 5. 引导性问题生成
- 根据文档内容自动生成引导性问题，帮助用户更好地提问

## 技术架构

### 核心组件

1. **知识库管理模块**（KnowledgeBase）
   - 文档处理和解析
   - 向量库的创建、加载和保存
   - 文件格式转换为Markdown

2. **检索模块**
   - BGE向量检索：使用HuggingFaceBgeEmbeddings进行语义检索
   - BM25关键词检索：基于jieba分词的传统关键词搜索
   - 文档重排序：使用FlagReranker优化检索结果

3. **API服务**（FastAPI）
   - 提供RESTful API接口
   - 支持文件上传、知识库操作和问答交互

4. **大语言模型集成**
   - 使用OpenAI兼容接口与大语言模型交互
   - 支持流式输出和参数调优

### 项目结构

```
Doc_QA/
├── Knowledge_based/          # 知识库存储目录
├── add/                      # 额外功能模块
│   ├── morefile/             # 更多文件格式处理
│   ├── ocr/                  # OCR文字识别功能
│   └── output-md-root/       # Markdown输出
├── images/                   # 图片存储
├── uploads/                  # 文件上传临时目录
├── app.py                    # 主API服务（旧版）
├── app2.py                   # 主API服务（新版）
├── functions.py              # 核心功能实现
├── Knowledge_base.py         # 知识库管理
├── bm25_search.py            # BM25搜索引擎
├── config.yaml               # 配置文件
└── req.txt                   # 依赖列表
```

## 安装指南

### 环境要求
- Python 3.8+
- CUDA 支持（推荐，用于加速模型推理）
- Windows 10/11 系统

### 安装依赖
```bash
pip install -r req.txt
```

### 注意事项
- vllm在Windows环境下不兼容，已在依赖文件中注释
- datrie库在Windows环境下需要安装Microsoft Visual C++ 14.0或更高版本，已在依赖文件中注释
- 确保安装Microsoft Visual C++ Redistributable以支持某些库
- 在Windows环境中，模型推理可能会比Linux慢，这是正常现象
- 如有路径问题，建议使用绝对路径避免Windows路径分隔符引起的问题
- 在Windows上使用时，某些模型推理功能可能需要替代方案

### 配置文件设置

编辑 `config.yaml` 文件，设置以下参数：

```yaml
paths:
  kb_dir: "你的知识库存储路径"
  model_dir: "BGE模型路径"
  reranker_model_dir: "重排序模型路径"
  openai_api_base: "大语言模型API地址"
  openai_api_keys: "API密钥"

models:
  embeddings_model: "bge-large-zh-v1.5"  # 嵌入模型
  reranker_model: "bge-reranker-large"    # 重排序模型
  llm_model: "deepseek-chat"             # 大语言模型

settings:
  batch_size: 1024
  device: "cuda"                         # 运行设备（cuda或cpu）
  normalize_embeddings: true
  use_fp16: true                         # 是否使用半精度
```

## 使用方法

### 启动服务

```bash
python app2.py
```

服务默认运行在 `http://127.0.0.1:7861`

### API接口

#### 1. 创建/更新知识库

```bash
POST /update_vectordb
Content-Type: multipart/form-data

kb_name: your_knowledge_base_name
files: 文件1, 文件2, ...
```

#### 2. 删除知识库

```bash
POST /delete_kb
Content-Type: application/x-www-form-urlencoded

kb_name: your_knowledge_base_name
```

#### 3. 文档问答

```bash
POST /mulitdoc_qa
Content-Type: application/json

{
  "model": "deepseek-chat",
  "messages": [
    {"role": "user", "content": "你的问题"}
  ],
  "temperature": 0.5,
  "stream": true,
  "only_chatKBQA": true,
  "kb_name": "your_knowledge_base_name",
  "show_source": true,
  "derivation": false
}
```

#### 4. 生成引导性问题

```bash
POST /view_guiding_questions
Content-Type: application/json

{
  "kb_name": "your_knowledge_base_name"
}
```

#### 5. 移除知识库中的文件

```bash
POST /remove_file
Content-Type: application/x-www-form-urlencoded

kb_name: your_knowledge_base_name
file_name: file_to_remove.pdf
```

## 核心功能实现

### 文档处理流程

1. **文件上传**：接收各种格式的文件
2. **格式转换**：将不同格式转换为统一的Markdown格式
3. **文本分割**：按段落或语义进行文本分割
4. **向量编码**：使用BGE模型将文本转换为向量
5. **向量存储**：使用FAISS构建向量索引并保存

### 检索与问答流程

1. **查询处理**：接收用户问题
2. **混合检索**：
   - 使用BGE向量模型检索相关文档
   - 使用BM25检索相关文档
3. **结果重排序**：使用BGE重排序模型优化检索结果
4. **上下文构建**：将检索到的文档构建为上下文
5. **生成回答**：发送给大语言模型生成最终回答

## 性能优化

- 使用多进程并行处理文档
- 支持批量向量编码
- 模型使用FP16加速推理（GPU环境下）
- 流式输出减少用户等待时间

## 注意事项

1. 确保配置文件中的模型路径正确且模型已下载
2. GPU环境下性能更好，建议优先使用GPU
3. 处理大型文档可能需要较长时间，请耐心等待
4. 如需修改端口或其他服务参数，请编辑app2.py中的相关配置

## 扩展方向

1. 添加更多文档格式支持
2. 实现更高效的向量索引和检索算法
3. 集成更多大语言模型选项
4. 开发Web前端界面
5. 添加用户权限管理

## 许可证

[MIT License](LICENSE)
## 不相干/无用文件清单（审计）

- 重复拷贝（与根目录实现重复，未被引用）：
  - `dana-knowledge/bm25_search.py`（重复 `bm25_search.py`）
  - `dana-knowledge/document_reranker.py`（重复 `document_reranker.py`）
  - `dana-knowledge/documen_processing.py`（重复 `documen_processing.py`）
  - `dana-knowledge/stopwords.txt`（重复 `stopwords.txt`）
  - `dana-knowledge/Knowledge_base.py`（旧实现，未被当前服务引用）
- 未使用模块/脚本（代码中无导入引用）：
  - `Knowledge_base.py`（根目录旧版，现用 `Knowledge_based_async.py`）
  - `embedding_api.py`
  - `file_processing_utils.py`
  - `async_functions.py`
  - `presentation.py`（根目录；当前使用的是 `add/morefile/presentation.py`）
  - `Qianwen_api2_test.py`
  - `add/morefile/try.ipynb`
  - `test_embedding_model_async.ipynb`
  - `add/morefile/requirement.txt`（与根目录 `req.txt` 重复）
  - 文档/临时产物：`output.md`、`r.md`、`ubuntu.txt`
- 示例与可选组件：
  - `Knowledge_based/广城理学生手册/`（示例知识库数据，可保留用于演示）
  - `server/index.html`（可选静态前端页面，未由后端路由直接提供，可手动打开）
  - `add/ocr/ocr_app.py`（可选 OCR 服务；`documen_processing.py:24` 使用的 `url_f` 指向外部服务，若不启用 OCR 则不必运行）

## 当前有效入口与核心接口

- 入口服务：`app.py`
- 主要接口位置（代码参考）：
  - 列举知识库：`app.py:123` `/list_kb`
  - 更新向量库（上传文件）：`app.py:156` `/update_vectordb`
  - 生成引导性问题：`app.py:345` `/view_guiding_questions`
  - 多文档问答（SSE/非SSE）：`app.py:435` `/mulitdoc_qa`
  - 删除知识库文件：`app.py:413` `/remove_file`
  - 汇总最终响应：`app.py:590` `/final_response`

## 项目结构快照（当前实际）

```
Doc_QA/
├── app.py
├── Knowledge_based_async.py
├── functions.py
├── bm25_search.py
├── document_reranker.py
├── documen_processing.py
├── config.yaml
├── stopwords.txt
├── server/
│   └── index.html
├── Knowledge_based/               # 示例知识库（含 faiss_index、markdown_directory、uploads）
├── add/
│   ├── morefile/
│   │   ├── ppt_processing.py
│   │   ├── html_processing.py
│   │   ├── excel_processing.py
│   │   ├── pic_processing.py
│   │   ├── doc_processing.py
│   │   ├── presentation.py
│   │   ├── rag/
│   │   └── deepdoc/
│   └── ocr/ocr_app.py
└── req.txt
```

说明：上方仅展示与当前服务相关的核心文件与目录，其余被标记为“未使用/重复”的文件建议后续清理或归档。
=======
# 文档问答系统（Doc_QA）

这是一个基于检索增强生成（RAG）技术的文档问答系统，支持多种文档格式的上传、解析和智能问答。系统结合向量检索和关键词检索，提供高效准确的文档信息检索和智能回答功能。

## 项目功能特点

### 1. 多格式文档支持
- 支持PDF、Word（.docx/.doc）、Markdown、TXT、PPTX、HTML、Excel、CSV等多种文档格式
- 支持图片（JPG、PNG）格式的OCR文字识别

### 2. 知识库管理
- 创建、删除和选择不同的知识库
- 每个知识库独立存储和管理文档

### 3. 混合检索策略
- 结合BGE向量检索和BM25关键词检索
- 使用BGE重排名模型优化检索结果

### 4. 智能问答功能
- 基于知识库内容进行上下文感知的回答
- 支持多轮对话和历史记录
- 流式输出回答结果
- 可选择显示回答来源和推理过程

### 5. 引导性问题生成
- 根据文档内容自动生成引导性问题，帮助用户更好地提问

## 技术架构

### 核心组件

1. **知识库管理模块**（KnowledgeBase）
   - 文档处理和解析
   - 向量库的创建、加载和保存
   - 文件格式转换为Markdown

2. **检索模块**
   - BGE向量检索：使用HuggingFaceBgeEmbeddings进行语义检索
   - BM25关键词检索：基于jieba分词的传统关键词搜索
   - 文档重排序：使用FlagReranker优化检索结果

3. **API服务**（FastAPI）
   - 提供RESTful API接口
   - 支持文件上传、知识库操作和问答交互

4. **大语言模型集成**
   - 使用OpenAI兼容接口与大语言模型交互
   - 支持流式输出和参数调优

## 环境要求

### 系统依赖
- Python 3.10+
- Windows/Linux/macOS
- NVIDIA GPU（推荐，用于加速模型推理）
- CUDA（如果使用GPU）

### Python依赖
主要依赖包括：
- langchain
- faiss-gpu（或faiss-cpu）
- transformers
- torch
- fastapi
- uvicorn
- python-multipart
- sentence-transformers
- FlagEmbedding
- jieba
- pymupdf
- python-docx
- openpyxl
- python-pptx
- beautifulsoup4
- pillow
- easyocr
- datrie

### 安装注意事项
- datrie库在Windows环境下需要安装Microsoft Visual C++ 14.0或更高版本，已在依赖文件中注释
- 确保安装Microsoft Visual C++ Redistributable以支持某些库
- 在Windows环境中，模型推理可能会比Linux慢，这是正常现象
- 如有路径问题，建议使用绝对路径避免Windows路径分隔符引起的问题
- 在Windows上使用时，某些模型推理功能可能需要替代方案

### 配置文件设置

编辑 `config.yaml` 文件，设置以下参数：

```yaml
paths:
  kb_dir: "你的知识库存储路径"
  model_dir: "BGE模型路径"
  reranker_model_dir: "重排序模型路径"
  openai_api_base: "大语言模型API地址"
  openai_api_keys: "API密钥"

models:
  embeddings_model: "bge-large-zh-v1.5"  # 嵌入模型
  reranker_model: "bge-reranker-large"    # 重排序模型
  llm_model: "deepseek-chat"             # 大语言模型

settings:
  batch_size: 1024
  device: "cuda"                         # 运行设备（cuda或cpu）
  normalize_embeddings: true
  use_fp16: true                         # 是否使用半精度
```

## 使用方法

### 启动服务

```bash
python app2.py
```

服务默认运行在 `http://127.0.0.1:7861`

### API接口

#### 1. 创建/更新知识库

```bash
POST /update_vectordb
Content-Type: multipart/form-data

kb_name: your_knowledge_base_name
files: 文件1, 文件2, ...
```

#### 2. 删除知识库

```bash
POST /delete_kb
Content-Type: application/x-www-form-urlencoded

kb_name: your_knowledge_base_name
```

#### 3. 文档问答

```bash
POST /mulitdoc_qa
Content-Type: application/json

{
  "model": "deepseek-chat",
  "messages": [
    {"role": "user", "content": "你的问题"}
  ],
  "temperature": 0.5,
  "stream": true,
  "only_chatKBQA": true,
  "kb_name": "your_knowledge_base_name",
  "show_source": true,
  "derivation": false
}
```

#### 4. 生成引导性问题

```bash
POST /view_guiding_questions
Content-Type: application/json

{
  "kb_name": "your_knowledge_base_name"
}
```

#### 5. 移除知识库中的文件

```bash
POST /remove_file
Content-Type: application/x-www-form-urlencoded

kb_name: your_knowledge_base_name
file_name: file_to_remove.pdf
```

## 核心功能实现

### 文档处理流程

1. **文件上传**：接收各种格式的文件
2. **格式转换**：将不同格式转换为统一的Markdown格式
3. **文本分割**：按段落或语义进行文本分割
4. **向量编码**：使用BGE模型将文本转换为向量
5. **向量存储**：使用FAISS构建向量索引并保存

### 检索与问答流程

1. **查询处理**：接收用户问题
2. **混合检索**：
   - 使用BGE向量模型检索相关文档
   - 使用BM25检索相关文档
3. **结果重排序**：使用BGE重排序模型优化检索结果
4. **上下文构建**：将检索到的文档构建为上下文
5. **生成回答**：发送给大语言模型生成最终回答

## 性能优化

- 使用多进程并行处理文档
- 支持批量向量编码
- 模型使用FP16加速推理（GPU环境下）
- 流式输出减少用户等待时间

## 注意事项

1. 确保配置文件中的模型路径正确且模型已下载
2. GPU环境下性能更好，建议优先使用GPU
3. 处理大型文档可能需要较长时间，请耐心等待
4. 如需修改端口或其他服务参数，请编辑app2.py中的相关配置

## 扩展方向

1. 添加更多文档格式支持
2. 实现更高效的向量索引和检索算法
3. 集成更多大语言模型选项
4. 开发Web前端界面
5. 添加用户权限管理

## 许可证

[MIT License](LICENSE)
## 冗余代码与优化建议

- 活跃后端入口：`app2.py`（提供 `/list_kb`、`/update_vectordb`、`/mulitdoc_qa` 等接口）。建议统一使用 `app2.py` 作为唯一服务入口。
- 重复模块（建议后续清理或归档）：
  - `functions copy.py`（重复 `functions.py`）
  - `dana-knowledge/bm25_search.py`（重复 `bm25_search.py`）
  - `dana-knowledge/document_reranker.py`（重复 `document_reranker.py`）
  - `dana-knowledge/documen_processing.py`（重复 `documen_processing.py`）
  - `app.py`（旧版服务，和 `app2.py` 功能重合）
- 代码内优化：
  - BM25 初始化移除 `tqdm` 进度条与无用打印，减少初始化与查询开销。
  - 在 `functions.get_top_documents` 中通过 `file_path` 去重，并限制进入重排的候选集合大小，降低重排模型负载。
  - 如需进一步优化，可考虑：
    - 将 `embeddings` 与 `reranker_model` 设为惰性加载（首次请求时初始化）。
    - BM25 支持增量更新以避免每次重建整个索引。
