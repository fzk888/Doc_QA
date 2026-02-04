import nest_asyncio
import os
import yaml
import logging
import asyncio
import requests
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_parse import LlamaParse
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
from langchain.schema import Document

# 加载环境变量
load_dotenv()

# 初始化日志
logger = logging.getLogger(__name__)

# 确保 nest_asyncio 已应用
nest_asyncio.apply()

# 全局变量，用于缓存解析器和LLM设置
_parser = None
_llm_initialized = False

def _init_llm():
    """初始化LLM设置（仅初始化一次）"""
    global _llm_initialized
    if not _llm_initialized:
        try:
            # 尝试从环境变量或config.yaml读取配置
            model = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
            base_url = os.getenv("DEEPSEEK_BASE_URL")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            
            # 如果环境变量中没有，尝试从config.yaml读取
            if not base_url or not api_key:
                try:
                    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
                    if os.path.exists(config_path):
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = yaml.safe_load(f)
                            paths = config.get("paths", {})
                            if not base_url:
                                base_url = paths.get("openai_api_base")
                            if not api_key:
                                api_key = paths.get("openai_api_keys")
                except Exception as e:
                    logger.warning(f"从config.yaml读取LLM配置失败: {e}")
            
            if base_url and api_key:
                Settings.llm = DeepSeek(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
                _llm_initialized = True
                logger.info("DeepSeek LLM 初始化成功")
            else:
                logger.warning("DeepSeek LLM 配置不完整，将不使用LLM进行智能分块")
                Settings.llm = None
                _llm_initialized = True  # 标记为已尝试，避免重复
        except Exception as e:
            logger.warning(f"初始化DeepSeek LLM失败: {e}")
            Settings.llm = None
            _llm_initialized = True  # 即使失败也标记为已初始化，避免重复尝试

def _get_parser():
    """获取或创建LlamaParse解析器（单例模式）"""
    global _parser
    if _parser is None:
        # 优先从config.yaml读取API key，如果没有则从环境变量读取
        api_key = None
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    api_key = config.get("paths", {}).get("llamaindex_api_key")
        except Exception as e:
            logger.warning(f"读取config.yaml失败: {e}")
        
        # 如果config中没有，则从环境变量读取
        if not api_key:
            api_key = os.getenv("LLAMAINDEX_API_KEY")
        
        if not api_key:
            raise ValueError("未找到LLAMAINDEX_API_KEY，请在config.yaml的paths.llamaindex_api_key中配置，或设置环境变量LLAMAINDEX_API_KEY")
        
        _parser = LlamaParse(
            api_key=api_key,
    result_type="markdown",
            verbose=False  # 改为False，减少日志输出
        )
    return _parser


def _extract_excel_images_ocr(file_path: str) -> str:
    """
    从 Excel 中提取内嵌图片并调用 OCR 服务识别文字。
    返回合并后的文字（可能为空字符串）。
    该步骤是增益手段，不影响主流程，失败时会被忽略。
    """
    import zipfile

    try:
        # 读取 OCR 服务 URL（与其它模块保持一致）
        ocr_url = None
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    _cfg = yaml.safe_load(f)
                    ocr_url = _cfg.get("paths", {}).get("ocr_service_url")
        except Exception as e:
            logger.warning(f"[Excel-OCR] 从config.yaml读取OCR地址失败: {e}")

        if not ocr_url:
            # 回退到默认地址（与 pic_processing.py / ocr_app.py 一致）
            ocr_url = "http://127.0.0.1:8001/detection_pic"

        texts = []
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in zf.namelist():
                lower = name.lower()
                # Excel 内嵌图片通常位于 xl/media/ 目录
                if lower.startswith("xl/media/") and lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    try:
                        data = zf.read(name)
                        files = {"file": (os.path.basename(name), data, "image/png")}
                        logger.info(f"[Excel-OCR] 发送图片到OCR: {name} url={ocr_url}")
                        resp = requests.post(ocr_url, files=files, timeout=20)
                        status = getattr(resp, "status_code", None)
                        resp.raise_for_status()
                        text = resp.json().get("detection_result", "") or ""
                        if text.strip():
                            texts.append(text.strip())
                            logger.info(f"[Excel-OCR] 图片 {name} OCR 成功，len={len(text)} status={status}")
                        else:
                            logger.info(f"[Excel-OCR] 图片 {name} OCR 返回空文本 status={status}")
                    except Exception as e:
                        logger.warning(f"[Excel-OCR] 处理图片 {name} 失败: {e}")

        if not texts:
            logger.info("[Excel-OCR] 未在Excel中识别到有效图片文字或无内嵌图片")
            return ""

        joined = "图片识别内容（来自Excel内嵌图片）\n" + "\n\n---\n\n".join(texts)
        logger.info(f"[Excel-OCR] 汇总图片OCR文本长度={len(joined)}")
        return joined

    except Exception as e:
        logger.warning(f"[Excel-OCR] 提取Excel图片并识别失败（不影响主流程）: {e}")
        return ""

async def process_excel_file(file_path, markdown_directory):
    """
    使用LlamaParse解析Excel文件（异步版本）
    
    Args:
        file_path: Excel文件路径
        markdown_directory: Markdown文件输出目录
        
    Returns:
        list: langchain Document对象列表
    """
    try:
        # 初始化LLM（如果需要）
        _init_llm()
        
        # 获取解析器
        parser = _get_parser()
        
        # 解析Excel文件（LlamaParse支持异步，但这里先使用同步方式）
        print(f"正在使用LlamaParse解析Excel文件: {os.path.basename(file_path)}")
        logger.info(f"正在使用LlamaParse解析Excel文件: {file_path}")
        # 在异步函数中运行同步的IO操作
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, parser.load_data, file_path)
        
        if not documents or len(documents) == 0:
            base_name = os.path.basename(file_path)
            raise ValueError(f"文件 {base_name} 解析后不包含有效内容。")
        
        print(f"解析完成，获得 {len(documents)} 个文档")
        
        # 使用MarkdownElementNodeParser进行智能分块
        print("正在使用MarkdownElementNodeParser切割数据...")
        logger.info("正在使用MarkdownElementNodeParser切割数据...")
        
        # 检查 LLM 是否已初始化
        if not _llm_initialized or Settings.llm is None:
            error_msg = "LLM 未初始化！MarkdownElementNodeParser 需要 LLM。"
            print(f"❌ 错误：{error_msg}")
            logger.error(error_msg)
            raise ValueError("LLM 未初始化。请确保配置了 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")
        
        node_parser = MarkdownElementNodeParser(
            llm=Settings.llm,
            num_workers=1  # 减少并发，避免资源竞争
        )
        # 在异步函数中运行同步操作
        nodes = await loop.run_in_executor(None, node_parser.get_nodes_from_documents, documents)
        # 用于构建 Excel 专用索引的节点列表（后续会把图片OCR内容也补进来）
        nodes_for_index = list(nodes) if nodes else []
        
        print(f"切割完成，共生成 {len(nodes)} 个节点。")
        logger.info(f"切割完成，共生成 {len(nodes)} 个节点。")
        
        if not nodes or len(nodes) == 0:
            error_msg = f"节点列表为空！无法继续处理文件 {os.path.basename(file_path)}"
            print(f"❌ 错误：{error_msg}")
            logger.error(f"节点列表为空！无法继续处理文件 {file_path}")
            raise ValueError(f"MarkdownElementNodeParser 未能生成任何节点，文件 {os.path.basename(file_path)} 可能无法正确解析")
        
        # 显示第一个节点信息（用于调试）
        if nodes:
            node_type = type(nodes[0]).__name__
            print(f"第一个节点类型: {node_type}")
            if hasattr(nodes[0], 'get_content'):
                preview = nodes[0].get_content()[:200] if nodes[0].get_content() else "空内容"
            elif hasattr(nodes[0], 'text'):
                preview = nodes[0].text[:200] if nodes[0].text else "空内容"
            else:
                preview = str(nodes[0])[:200]
            print(f"第一个节点内容预览（前200字符）: {preview}")
        
        # 将LlamaIndex的节点转换为langchain Document对象
        langchain_documents = []
        for i, node in enumerate(nodes):
            try:
                # 提取节点内容 - 尝试多种方法，优先使用 get_content()
                content = None
                
                # 方法1: 使用 get_content() (推荐方法)
                if hasattr(node, 'get_content'):
                    try:
                        content = node.get_content()
                    except Exception as e:
                        logger.debug(f"节点 {i} get_content() 失败: {e}")
                
                # 方法2: 使用 text 属性
                if not content and hasattr(node, 'text'):
                    content = node.text
                
                # 方法3: 使用 node.text (嵌套节点)
                if not content and hasattr(node, 'node') and hasattr(node.node, 'text'):
                    content = node.node.text
                
                # 方法4: 使用 get_text()
                if not content and hasattr(node, 'get_text'):
                    try:
                        content = node.get_text()
                    except Exception:
                        pass
                
                # 方法5: 转换为字符串
                if not content:
                    content = str(node)
                
                # 确保内容是字符串类型
                content = str(content) if content is not None else ""
                
                # 检查内容是否为空
                if not content or len(content.strip()) == 0:
                    logger.warning(f"节点 {i} 内容为空，跳过")
                    print(f"⚠️ 警告：节点 {i} 内容为空，跳过")
                    continue
                
                # 打印节点内容长度（用于调试）
                content_length = len(content)
                print(f"节点 {i} 内容长度: {content_length} 字符")
                if content_length < 50:
                    print(f"⚠️ 警告：节点 {i} 内容很短，可能不完整: {content[:100]}")
                
                # 提取元数据
                metadata = {}
                if hasattr(node, 'metadata') and node.metadata:
                    metadata = dict(node.metadata)  # 确保是字典类型
                elif hasattr(node, 'node') and hasattr(node.node, 'metadata') and node.node.metadata:
                    metadata = dict(node.node.metadata)
                
                # 确保 file_path 在元数据中
                metadata["file_path"] = file_path
                
                # 创建langchain Document
                doc = Document(
                    page_content=content,  # 直接使用 content，不需要 str() 转换
                    metadata=metadata
                )
                langchain_documents.append(doc)
                
            except Exception as e:
                error_msg = f"转换节点 {i} 时出错: {e}"
                logger.warning(error_msg)
                print(f"⚠️ 警告：{error_msg}，跳过该节点")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        if len(langchain_documents) == 0:
            error_msg = f"未能成功转换任何节点为 langchain Document！"
            print(f"❌ 错误：{error_msg}")
            logger.error(error_msg)
            raise ValueError(f"无法将节点转换为 langchain Document，文件 {os.path.basename(file_path)} 处理失败")
        
        # 验证和统计返回的 Document
        total_chars = 0
        valid_docs = []
        for i, doc in enumerate(langchain_documents):
            content_len = len(doc.page_content) if doc.page_content else 0
            total_chars += content_len
            
            # 验证 Document 格式
            if not hasattr(doc, 'page_content'):
                logger.warning(f"Document {i} 缺少 page_content 属性")
                continue
            if not hasattr(doc, 'metadata'):
                logger.warning(f"Document {i} 缺少 metadata 属性")
                continue
            if 'file_path' not in doc.metadata:
                logger.warning(f"Document {i} metadata 中缺少 file_path")
                doc.metadata['file_path'] = file_path
            
            valid_docs.append(doc)
        
        print(f"成功转换 {len(valid_docs)} 个节点为 langchain Document")
        print(f"总字符数: {total_chars}, 平均每个文档: {total_chars // len(valid_docs) if valid_docs else 0} 字符")
        logger.info(f"成功转换 {len(valid_docs)} 个节点为 langchain Document，总字符数: {total_chars}")

        # Excel 内嵌图片 OCR：作为额外文档加入
        try:
            ocr_text = _extract_excel_images_ocr(file_path)
            if ocr_text and ocr_text.strip():
                ocr_doc = Document(
                    page_content=ocr_text,
                    metadata={
                        "file_path": file_path,
                        "source": "excel_image_ocr"
                    }
                )
                valid_docs.append(ocr_doc)
                print(f"Excel 图片OCR内容已加入文档列表，长度 {len(ocr_text)} 字符")
                logger.info(f"Excel 图片OCR内容已加入文档列表，长度 {len(ocr_text)} 字符")
                # 同时把 OCR 文本加入 LlamaIndex 索引节点里，确保 retriever 能检索到这段内容
                try:
                    from llama_index.core.schema import TextNode
                    nodes_for_index.append(
                        TextNode(
                            text=ocr_text,
                            metadata={"file_path": file_path, "source": "excel_image_ocr"},
                        )
                    )
                except Exception as e:
                    logger.warning(f"将图片OCR内容转换为TextNode失败（将导致LlamaIndex检索不到OCR文本）: {e}")
        except Exception as e:
            print(f"⚠️ 警告：Excel 图片OCR处理失败（忽略，不影响主流程）: {e}")
            logger.warning(f"Excel 图片OCR处理失败: {e}")
        
        # 显示前几个文档的内容预览
        for i, doc in enumerate(valid_docs[:3]):
            preview = doc.page_content[:200] if doc.page_content else "空内容"
            print(f"文档 {i} 预览（前200字符）: {preview}")
        
        # 可选：保存Markdown文件到指定目录（用于调试或查看）
        if markdown_directory and os.path.exists(markdown_directory):
            try:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                markdown_file = os.path.join(markdown_directory, f"{base_name}_llamaparse.md")
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    # 将所有文档内容合并写入
                    for doc in valid_docs:
                        f.write(doc.page_content)
                        f.write("\n\n---\n\n")
                logger.info(f"Markdown文件已保存到: {markdown_file}")
            except Exception as e:
                logger.warning(f"保存Markdown文件失败: {e}")
        
        # 构建并保存LlamaIndex索引（用于Excel专用查询，不走重排逻辑）
        try:
            # 获取知识库目录（从markdown_directory推断）
            # markdown_directory通常是: Knowledge_based/{kb_name}/markdown_directory
            # 所以 kb_dir 应该是: Knowledge_based/{kb_name}
            kb_dir = os.path.dirname(markdown_directory) if markdown_directory else None
            if kb_dir and os.path.exists(kb_dir):
                # 使用绝对路径，避免相对路径问题
                kb_dir = os.path.abspath(kb_dir)
                chroma_db_path = os.path.join(kb_dir, "chroma_db_excel")
                os.makedirs(chroma_db_path, exist_ok=True)
                
                # 从config.yaml读取模型路径
                model_path = None
                try:
                    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
                    if os.path.exists(config_path):
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = yaml.safe_load(f)
                            model_path = config.get("paths", {}).get("model_dir", r"D:\大模型应用开发\RAG\Doc_QA\model\bge-large-zh-v1.5")
                except Exception as e:
                    logger.warning(f"读取config.yaml失败: {e}")
                    model_path = r"D:\大模型应用开发\RAG\Doc_QA\model\bge-large-zh-v1.5"
                
                embed_model = HuggingFaceEmbedding(
                    model_name=model_path,
                    device='cpu',
                    normalize=True
                )
                
                # 初始化ChromaDB（使用知识库特定的路径）
                db_client = chromadb.PersistentClient(path=chroma_db_path)
                # 使用知识库名称作为collection名称的一部分，避免冲突
                kb_name = os.path.basename(kb_dir)
                collection_name = f"excel_rag_{kb_name}"
                chroma_collection = db_client.get_or_create_collection(collection_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # 构建索引（使用之前解析得到的nodes）
                print(f"正在构建LlamaIndex索引（用于Excel专用查询）...")
                print(f"知识库目录: {kb_dir}")
                print(f"索引保存路径: {chroma_db_path}")
                logger.info(f"正在构建LlamaIndex索引（用于Excel专用查询），知识库: {kb_name}, 路径: {chroma_db_path}")
                index = VectorStoreIndex(
                    nodes_for_index,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
                
                print(f"LlamaIndex索引构建成功，已保存到: {chroma_db_path}")
                logger.info(f"LlamaIndex索引构建成功，已保存到: {chroma_db_path}, 知识库: {kb_name}")
        except Exception as e:
            logger.warning(f"构建LlamaIndex索引失败（不影响主流程）: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # 返回验证后的文档列表
        return valid_docs
        
    except Exception as e:
        logger.error(f"解析Excel文件失败 {file_path}: {e}")
        raise


# Excel文档专用的查询函数（使用LlamaIndex的查询引擎，不走重排逻辑）
_excel_indices = {}  # 缓存已构建的索引，key为知识库名称

def _get_excel_index_for_kb(kb_name, kb_dir):
    """
    获取或构建Excel文档的LlamaIndex索引
    
    Args:
        kb_name: 知识库名称
        kb_dir: 知识库目录
        
    Returns:
        VectorStoreIndex: LlamaIndex的索引对象，如果不存在则返回None
    """
    global _excel_indices
    
    # 检查缓存
    if kb_name in _excel_indices:
        return _excel_indices[kb_name]
    
    try:
        # 检查是否存在ChromaDB索引（使用绝对路径）
        kb_dir = os.path.abspath(kb_dir) if kb_dir else None
        if not kb_dir or not os.path.exists(kb_dir):
            logger.warning(f"知识库目录不存在: {kb_dir}")
            return None
        
        chroma_db_path = os.path.join(kb_dir, "chroma_db_excel")
        logger.info(f"检查Excel索引路径: {chroma_db_path}")
        if not os.path.exists(chroma_db_path):
            logger.warning(f"Excel索引不存在: {chroma_db_path}，需要重新构建")
            return None
        
        # 从config.yaml读取模型路径
        model_path = None
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    model_path = config.get("paths", {}).get("model_dir", r"D:\大模型应用开发\RAG\Doc_QA\model\bge-large-zh-v1.5")
        except Exception as e:
            logger.warning(f"读取config.yaml失败: {e}")
            model_path = r"D:\大模型应用开发\RAG\Doc_QA\model\bge-large-zh-v1.5"
        
        embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            device='cpu',
            normalize=True
        )
        
        # 加载ChromaDB（使用与构建时相同的collection名称）
        db_client = chromadb.PersistentClient(path=chroma_db_path)
        collection_name = f"excel_rag_{kb_name}"
        chroma_collection = db_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 加载索引
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        _excel_indices[kb_name] = index
        logger.info(f"成功加载Excel索引: {kb_name}")
        return index
        
    except Exception as e:
        logger.error(f"加载Excel索引失败 {kb_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def query_excel_with_llamaindex(query: str, kb_name: str, kb_dir: str, req_id=None):
    """
    使用LlamaIndex查询引擎查询Excel文档（不走重排逻辑）
    
    Args:
        query: 查询语句
        kb_name: 知识库名称
        kb_dir: 知识库目录
        req_id: 请求ID（用于日志）
        
    Returns:
        list: [(Document, score)] 格式的列表，score固定为1.0（因为不走重排）
    """
    _pref = f"[req:{req_id}] " if req_id else ""
    
    try:
        # 获取或构建索引
        index = _get_excel_index_for_kb(kb_name, kb_dir)
        if index is None:
            logger.warning(f"{_pref}Excel索引不存在，无法使用LlamaIndex查询")
            return []
        
        # 关键：Excel 这里不要走 as_query_engine()，否则会触发 LlamaIndex 默认 OpenAI LLM（无 KEY 会报错）
        # 我们只做“检索”，不做“生成”，用 retriever 即可
        logger.info(f"{_pref}使用LlamaIndex retriever 查询Excel文档: {query}")
        try:
            # top_k 取大一些：OCR 节点通常更短，向量相似度可能被长表格摘要压过
            retriever = index.as_retriever(similarity_top_k=10)
            retrieved_nodes = retriever.retrieve(query)
            # retrieved_nodes 可能是 NodeWithScore
            source_nodes = []
            for nw in retrieved_nodes or []:
                if hasattr(nw, "node"):
                    source_nodes.append(nw.node)
                else:
                    source_nodes.append(nw)
            logger.info(f"{_pref}retriever 检索到 {len(source_nodes)} 个节点")
        except Exception as e:
            logger.warning(f"{_pref}retriever 检索失败: {e}")
            source_nodes = []
        
        # 将LlamaIndex的节点转换为langchain Document
        langchain_docs = []
        for node in source_nodes:
            try:
                # 提取内容
                content = None
                if hasattr(node, 'get_content'):
                    content = node.get_content()
                elif hasattr(node, 'text'):
                    content = node.text
                elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                    content = node.node.text
                else:
                    content = str(node)
                
                if not content:
                    continue
                
                # 提取元数据
                metadata = {}
                if hasattr(node, 'metadata') and node.metadata:
                    metadata = dict(node.metadata)
                elif hasattr(node, 'node') and hasattr(node.node, 'metadata') and node.node.metadata:
                    metadata = dict(node.node.metadata)
                
                # 兜底：确保 file_path 存在，避免后续 selected_docs 为空
                if 'file_path' not in metadata or not metadata.get('file_path'):
                    metadata['file_path'] = os.path.join(kb_dir, "unknown.xlsx") if kb_dir else "unknown.xlsx"
                # 标记来源
                if 'source' not in metadata:
                    metadata['source'] = 'excel_llamaindex'
                
                from langchain.schema import Document
                doc = Document(page_content=str(content), metadata=metadata)
                langchain_docs.append((doc, 1.0))  # 固定得分为1.0，因为不走重排
            except Exception as e:
                logger.warning(f"{_pref}转换节点失败: {e}")
                continue

        # 控制返回数量，避免 prompt 过长
        langchain_docs = langchain_docs[:3]

        logger.info(f"{_pref}LlamaIndex查询Excel文档，返回 {len(langchain_docs)} 个结果")
        if langchain_docs:
            logger.info(f"{_pref}第一个结果预览: {langchain_docs[0][0].page_content[:200]}")
        return langchain_docs
        
    except Exception as e:
        logger.error(f"{_pref}LlamaIndex查询Excel文档失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("用法: python add/morefile/excel.py <excel_file_path>")
        sys.exit(0)

    test_file = os.path.abspath(" ".join(sys.argv[1:]))
    if not os.path.exists(test_file):
        print(f"❌ 错误：文件不存在: {test_file}")
        sys.exit(1)

    async def _main():
        md_dir = os.path.join(os.getcwd(), "test_markdown_excel")
        os.makedirs(md_dir, exist_ok=True)
        docs = await process_excel_file(test_file, md_dir)
        print(f"完成：返回 {len(docs)} 个 Document")
        for i, d in enumerate(docs[:3]):
            print(f"[{i}] len={len(d.page_content)} meta_keys={list(d.metadata.keys())}")

    asyncio.run(_main())