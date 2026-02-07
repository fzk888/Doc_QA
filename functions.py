import os
import json
import shutil
import logging
import random
import re
import uuid
from collections import OrderedDict
import aiofiles
import tiktoken
import yaml
from typing import List
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from core.kb_manager import KnowledgeBase
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from core.reranker import DocumentReranker
from core.search_bm25 import BM25Search
import time
import random
from concurrent.futures import ThreadPoolExecutor
# Load .env (API keys, base urls, etc.)
# 默认从当前工作目录（项目根目录）读取 .env；如需自定义路径可设置 DOTENV_PATH
load_dotenv(os.getenv("DOTENV_PATH"))

# Load configuration
# 从项目根目录加载全局配置（模型、阈值、路径等），后续各模块统一读取
with open("config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

def _get_env_first(*keys: str) -> str | None:
    """
    依次尝试从环境变量读取；返回第一个非空值。
    """
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return v
    return None

# --- Secrets / endpoints: env overrides config ---
# 兼容历史字段：
# - config.paths.openai_api_keys / openai_api_base
# - 以及 add/morefile/excel.py 中用到的 DEEPSEEK_* 变量
_api_key = _get_env_first("OPENAI_API_KEY", "OPENAI_API_KEYS", "DEEPSEEK_API_KEY")
_base_url = _get_env_first("OPENAI_API_BASE", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")
if "paths" not in config or config["paths"] is None:
    config["paths"] = {}
if _api_key:
    config["paths"]["openai_api_keys"] = _api_key
if _base_url:
    config["paths"]["openai_api_base"] = _base_url

# Configure logging
logger = logging.getLogger("docqa")
logger.setLevel(logging.INFO)
logger.propagate = False

LOG_PROMPT_MAX_CHARS = int(os.getenv("LOG_PROMPT_MAX_CHARS", "300"))
# 安全日志：仅记录提示词摘要，避免在日志中写入超长文本或用户隐私
def _log_prompt(text):
    if len(text) <= LOG_PROMPT_MAX_CHARS:
        logger.info(f"Prompt: {text}")
    else:
        logger.info(f"Prompt length: {len(text)} chars (truncated)")
        logger.info(f"Prompt start: {text[:LOG_PROMPT_MAX_CHARS]}...")
        logger.info(f"Prompt end: ...{text[-LOG_PROMPT_MAX_CHARS:]}")

def _summarize_docs(docs):
    try:
        names = [os.path.basename(doc.metadata.get('file_path', '')) for doc in docs]
    except Exception:
        names = []
    n = len(docs) if docs is not None else 0
    head = names[:3]
    extra = n - len(head)
    return f"docs_count={n} sources={head}" + (f" (+{extra} more)" if extra > 0 else "")
def _truncate_text(text, max_chars):
    return text if len(text) <= max_chars else text[:max_chars] + f"...(truncated {len(text) - max_chars} chars)"
def _docs_preview(docs, max_docs=3, preview_chars=300):
    try:
        items = []
        for i, d in enumerate(docs[:max_docs]):
            name = os.path.basename(d.metadata.get('file_path', ''))
            preview = _truncate_text(d.page_content, preview_chars)
            items.append({"source": name, "preview": preview})
        more = len(docs) - len(items)
        return str(items) + (f" (+{more} more)" if more > 0 else "")
    except Exception:
        return "[]"
def _render_prompt_safe(prompt, **kwargs):
    safe = dict(kwargs)
    if 'top_documents' in safe and safe['top_documents'] is not None:
        if isinstance(safe['top_documents'], (list, tuple)):
            safe['top_documents'] = _docs_preview(safe['top_documents'])
        else:
            safe['top_documents'] = _truncate_text(str(safe['top_documents']), 800)
    if 'document' in safe:
        doc_val = safe['document']
        if isinstance(doc_val, list):
            doc_str = "\n".join([str(x) for x in doc_val]) 
        else:
            doc_str = str(doc_val)
        safe['document'] = _truncate_text(doc_str, 800)
    if 'uploaded_files' in safe and safe['uploaded_files'] is not None:
        try:
            sample = list(sorted(safe['uploaded_files']))[:5]
        except Exception:
            sample = []
        safe['uploaded_files'] = str(sample)
    if 'history' in safe and safe['history'] is not None:
        safe['history'] = _truncate_text(safe['history'], 800)
    if 'history_str' in safe and safe['history_str'] is not None:
        safe['history_str'] = _truncate_text(safe['history_str'], 800)
    if 'query' in safe and safe['query'] is not None:
        safe['query'] = _truncate_text(str(safe['query']), 500)
    try:
        return prompt.format(**safe)
    except Exception:
        return str(safe)

model_kwargs = {"device": config['settings']['device']}
encode_kwargs = {
    "batch_size": config['settings']['batch_size'],
    "normalize_embeddings": config['settings']['normalize_embeddings']
}
_embeddings = None
_reranker_model = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("正在初始化嵌入模型（BGE）...")
        _embeddings = HuggingFaceBgeEmbeddings(
            model_name=config['paths']['model_dir'],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return _embeddings

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        try:
            _reranker_model = FlagReranker(
                config['paths']['reranker_model_dir'],
                use_fp16=config['settings'].get('use_fp16', True),
            )
        except Exception as e:
            # Windows 上常见：OSError 1455（页面文件太小/虚拟内存不足）导致模型无法加载
            logger.exception(f"重排模型加载失败，将禁用重排并降级到非重排检索: {e}")
            _reranker_model = False  # sentinel：表示已尝试但失败
    return _reranker_model


class KBState:
    def __init__(self):
        # 当前知识库对象（封装加载/更新向量库等）
        self.kb = None
        # FAISS 向量库句柄（为空表示尚未构建或加载失败）
        self.kb_vectordb = None
        # 当前选中的知识库名称
        self.current_kb_name = None
        # 简易对话历史（用于拼接到提示词或检索改写）
        self.history = []
        # 未过滤的上下文集合（用于 BM25 检索器构建）
        self.unfilter_context = []
        # 基于未过滤上下文的 BM25 检索器
        self.searcher_from_target_doc = None

kb_state = KBState()
    
def get_top_documents(query: str, req_id=None):
    # 融合检索入口：并行执行向量检索 + BM25，去重后使用重排模型计算相关分，返回 [(Document, score)]
    # 注意：需要先确保 kb_state.kb_vectordb / searcher_from_target_doc 可用
    _pref = f"[req:{req_id}] " if req_id else ""
    logger.debug(f"{_pref}KB: {kb_state.current_kb_name}")
    
    # 特殊处理：根据配置决定Excel文档的查询策略
    if kb_state.kb_vectordb and hasattr(kb_state.kb_vectordb, 'docstore'):
        excel_docs = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items() 
                     if doc.metadata.get('file_path', '').endswith(('.xlsx', '.xls', '.xlsm', '.csv'))]
        total_docs = len(kb_state.kb_vectordb.docstore._dict)
        
        # 读取配置中的Excel查询策略
        excel_strategy = config.get('settings', {}).get('excel_query_strategy', 'auto')
        
        # 决定是否使用LlamaIndex查询
        use_llamaindex = False
        if excel_strategy == 'llamaindex':
            # 强制使用LlamaIndex（只要有Excel文档）
            use_llamaindex = len(excel_docs) > 0
            if use_llamaindex:
                logger.info(f"{_pref}配置为强制使用LlamaIndex查询（检测到 {len(excel_docs)} 个Excel文档）")
        elif excel_strategy == 'faiss':
            # 强制使用FAISS（常规检索）
            use_llamaindex = False
            logger.info(f"{_pref}配置为使用FAISS查询（常规检索+重排）")
        elif excel_strategy == 'auto':
            # 自动检测：如果所有文档都是Excel，使用LlamaIndex；否则使用FAISS
            use_llamaindex = (total_docs > 0 and len(excel_docs) == total_docs)
            if use_llamaindex:
                logger.info(f"{_pref}自动检测：知识库只包含Excel文档（{total_docs}个），使用LlamaIndex查询引擎")
            else:
                logger.info(f"{_pref}自动检测：知识库包含混合文档（Excel: {len(excel_docs)}/{total_docs}），使用FAISS查询（常规检索+重排）")
        
        # 如果决定使用LlamaIndex查询
        if use_llamaindex:
            try:
                from add.morefile.excel import query_excel_with_llamaindex
                import yaml
                
                # 获取知识库目录（使用局部变量避免覆盖全局 config）
                with open("config.yaml", "r", encoding="utf-8") as config_file:
                    local_cfg = yaml.safe_load(config_file)
                KB_DIR = local_cfg['paths']['kb_dir']
                kb_dir = os.path.join(KB_DIR, kb_state.current_kb_name)
                
                # 使用LlamaIndex查询
                excel_results = query_excel_with_llamaindex(query, kb_state.current_kb_name, kb_dir, req_id=req_id)
                if excel_results:
                    logger.info(f"{_pref}LlamaIndex查询成功，返回 {len(excel_results)} 个结果")
                    return excel_results
                else:
                    logger.warning(f"{_pref}LlamaIndex查询未返回结果，回退到常规检索")
            except Exception as e:
                logger.warning(f"{_pref}LlamaIndex查询失败，回退到常规检索: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    
    if not kb_state.kb_vectordb or not kb_state.searcher_from_target_doc:
        raise ValueError("Knowledge base not loaded.")

    # 调试：检查向量库中的文档数量和内容
    if kb_state.kb_vectordb and hasattr(kb_state.kb_vectordb, 'docstore'):
        total_docs = len(kb_state.kb_vectordb.docstore._dict)
        logger.info(f"{_pref}向量库中共有 {total_docs} 个文档块")
        # 检查是否有Excel相关的文档
        excel_docs = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items() 
                     if doc.metadata.get('file_path', '').endswith(('.xlsx', '.xls', '.xlsm', '.csv'))]
        logger.info(f"{_pref}其中Excel相关文档: {len(excel_docs)} 个")
        if excel_docs:
            for i, doc in enumerate(excel_docs[:3]):
                content_preview = doc.page_content[:200] if doc.page_content else "空"
                logger.info(f"{_pref}Excel文档 {i}: file_path={os.path.basename(doc.metadata.get('file_path', ''))}, "
                          f"content_len={len(doc.page_content) if doc.page_content else 0}, "
                          f"preview={content_preview}")
    
    # 调试：检查BM25索引中的文档
    if kb_state.searcher_from_target_doc and hasattr(kb_state.searcher_from_target_doc, 'docs'):
        bm25_docs_count = len(kb_state.searcher_from_target_doc.docs)
        logger.info(f"{_pref}BM25索引中共有 {bm25_docs_count} 个文档")
        excel_bm25 = [doc for doc in kb_state.searcher_from_target_doc.docs 
                     if doc.metadata.get('file_path', '').endswith(('.xlsx', '.xls', '.xlsm', '.csv'))]
        logger.info(f"{_pref}BM25中Excel相关文档: {len(excel_bm25)} 个")

    # 适度降低向量检索返回数量，减少后续重排负载
    logger.debug(f"{_pref}Performing vector and BM25 search...")
    # 限制向量检索返回数量，降低后续重排负载
    retriever = kb_state.kb_vectordb.as_retriever(search_kwargs={"k": 4})
    # 并行执行向量检索与 BM25 检索以降低总体等待时间
    # 并行执行两路检索，缩短端到端等待时间（BGE 向量检索 + BM25）
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_bge = ex.submit(retriever.invoke, query)
        logger.debug(f"{_pref}BM25 search started")
        fut_bm25 = ex.submit(kb_state.searcher_from_target_doc.search, query, 0.2)
        bge_context = fut_bge.result()
        bm25_context = fut_bm25.result()
    try:
        logger.info(f"{_pref}retrieval bge={len(bge_context)} bm25={len(bm25_context)}")
        # 调试：显示检索到的文档内容预览
        if bge_context:
            logger.info(f"{_pref}BGE检索到的文档预览: {[os.path.basename(d.metadata.get('file_path', '')) for d in bge_context[:3]]}")
        if bm25_context:
            logger.info(f"{_pref}BM25检索到的文档预览: {[os.path.basename(d.metadata.get('file_path', '')) for d in bm25_context[:3]]}")
    except Exception:
        pass
    # BM25 已经按得分排序，限制候选数量避免过多文档进入重排
    # BM25 已按得分排序；控制候选上限，避免过多文档进入重排
    if len(bm25_context) > 8:
        bm25_context = bm25_context[:8]

    logger.debug(f"{_pref}Fusing search results...")
    # 融合两路检索结果；后续去重
    merged_res = bge_context + bm25_context

    if len(merged_res) == 0:
        logger.info(f"{_pref}retrieval merged=0")
        return []

    # 基于内容去重，避免重复的文档块
    # 使用 (file_path, content_hash) 作为唯一标识，这样同一文件的不同chunk都能保留
    unique_docs_dict = {}
    for doc in merged_res:
        file_path = doc.metadata.get('file_path', '')
        # 使用内容的前100字符作为简单hash（避免对完整内容做hash计算）
        content_preview = doc.page_content[:100] if doc.page_content else ''
        unique_key = (file_path, hash(content_preview))
        # 如果这个key不存在，或者当前文档内容更长（保留更完整的内容）
        if unique_key not in unique_docs_dict or len(doc.page_content) > len(unique_docs_dict[unique_key].page_content):
            unique_docs_dict[unique_key] = doc
    unique_docs = list(unique_docs_dict.values())
    try:
        logger.info(f"{_pref}unique_docs={len(unique_docs)}")
    except Exception:
        pass

    if len(unique_docs) == 1:
        # 单候选仍走重排获取真实分值，避免固定分导致误判相关
        reranker = DocumentReranker(get_reranker_model())
        candidates = unique_docs[:1]
        top_documents_with_scores = reranker.rerank_documents(query, candidates, top_n=1)
        return [(doc, round(score, 2)) for doc, score in top_documents_with_scores if doc.metadata.get('file_path')]

    logger.debug(f"{_pref}Reranking documents...")
    # 使用重排模型对候选进行相关性打分，选出 Top-N
    _rm = get_reranker_model()
    if _rm is False:
        # 降级：不做重排，直接返回候选（保留原始顺序/相似度顺序）
        # 这里给一个较低但非0的分，避免后续严格KB阈值直接判定“未命中”
        return [(doc, 0.2) for doc in candidates[:3] if doc.metadata.get('file_path')]
    reranker = DocumentReranker(_rm)
    # 控制进入重排的最大候选数量
    candidates = unique_docs[:8]
    
    # 过滤掉内容太短的文档（少于50字符的文档通常没有实际内容）
    min_content_length = 50
    filtered_candidates = [doc for doc in candidates if doc.page_content and len(doc.page_content.strip()) >= min_content_length]
    if len(filtered_candidates) < len(candidates):
        removed_count = len(candidates) - len(filtered_candidates)
        logger.info(f"{_pref}过滤掉 {removed_count} 个内容过短的文档（少于{min_content_length}字符）")
        candidates = filtered_candidates
    
    if not candidates:
        logger.warning(f"{_pref}所有候选文档都被过滤，无法进行重排")
        return []
    
    try:
        _cand_names = [os.path.basename(d.metadata.get('file_path', '')) for d in candidates if d.metadata.get('file_path')]
        logger.info(f"{_pref}candidates={_cand_names} (共{len(candidates)}个)")
        # 调试：显示候选文档的内容预览
        for i, doc in enumerate(candidates[:3]):
            content_preview = doc.page_content[:200] if doc.page_content else "空"
            logger.info(f"{_pref}候选文档 {i} 内容预览: {content_preview}")
    except Exception:
        pass
    
    # 调试：显示查询内容
    logger.info(f"{_pref}重排查询: {query}")
    top_documents_with_scores = reranker.rerank_documents(query, candidates, top_n=3)
    
    # 调试：显示重排得分详情
    try:
        logger.info(f"{_pref}重排得分详情（原始值）: {[(os.path.basename(doc.metadata.get('file_path', '')), round(score, 4)) for doc, score in top_documents_with_scores]}")
    except Exception:
        pass
    try:
        _sel_names = [os.path.basename(doc.metadata.get('file_path', '')) for doc, _ in top_documents_with_scores if doc.metadata.get('file_path')]
        _sel_scores = [round(score, 2) for _, score in top_documents_with_scores if _]
        logger.info(f"{_pref}rerank_top={_sel_names} scores={_sel_scores}")
    except Exception:
        pass

    # 返回 (Document, score) 且要求存在 file_path 元数据
    # 注意：这里保留原始得分（不四舍五入），让后续的阈值判断更准确
    return [(doc, score) for doc, score in top_documents_with_scores if doc.metadata.get('file_path')]

def run_llm_Knowlege_baes_file_QA(query: str, keep_history: bool = True):
    # 基于“已上传文件列表”的简单 KB QA（不做分段检索），用于文件级预览与说明
    openai_api_key = config['paths']['openai_api_keys']
    openai_api_base = config['paths']['openai_api_base']

    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=True
    )
    logger.debug(f"LLM model: {config['models']['llm_model']}")

    # 将历史拼接到提示词，提供上下文参考
    history_str = "\n".join([str(item) for item in kb_state.history]) + "\n这是以上我和你的对话记录，请参考\n"
    # use global kb_state.kb_vectordb; fall back to empty set if not available
    if getattr(kb_state, 'kb_vectordb', None) is None:
        uploaded_files = set()
    else:
        uploaded_files = {os.path.basename(doc.metadata.get('file_path', '')) for doc in kb_state.kb_vectordb.docstore._dict.values()}
    prompt_template = "以上是历史信息{history_str}，您是一位大型语言人工智能助手。您将被提供一个用户问题,根据知识库文档列表{uploaded_files},结合问题{query}，撰写一个清晰、简洁且准确的答案。回答："
    prompt = PromptTemplate(template=prompt_template, input_variables=["history_str", "uploaded_files", "query"])
    rendered_safe = _render_prompt_safe(prompt, history_str=history_str, uploaded_files=uploaded_files, query=query)
    _log_prompt(rendered_safe)
    rag_chain = (
        {"history_str": lambda x: history_str, "uploaded_files": lambda x: uploaded_files, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response_text = ""
    for chunk in rag_chain.stream(query):
        response_text += chunk
        yield chunk

    if keep_history:
        kb_state.history.append({"query": query, "response": response_text})

def find_image_links(documents):
    image_info = []
    for doc in documents:
        matches = re.findall(r'!\[.*?\]\((.*?)\)', doc.page_content)
        image_info.extend(matches)
    return image_info



#使用 OpenAI API 来生成引导性问题。这个函数将遍历知识库中的每个文档，并生成一个与文档内容相关的引导性问题。


def generate_guiding_questions(num_questions_total=3, num_questions_per_doc=2):
    # 遍历知识库文档，调用 LLM 生成若干条引导性问题以辅助用户发问
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=True
    )
    logger.debug(f"LLM model: {config['models']['llm_model']}")
    # global kb_state
    kb_vectordb = kb_state.kb_vectordb
    doc_groups = {}
    for doc_id, doc in kb_vectordb.docstore._dict.items():
        source = doc.metadata.get('source', 'unknown')
        if source not in doc_groups:
            doc_groups[source] = []
        doc_groups[source].append(doc)
    
    guiding_questions = []
    
    # 如果只有一个文档组，且需要的问题数量大于每个文档的问题数量
    if len(doc_groups) == 1 and num_questions_total > num_questions_per_doc:
        single_source = list(doc_groups.keys())[0]
        docs = doc_groups[single_source]
        
        for i in range(num_questions_total):
            doc = random.choice(docs)  # 随机选择一个文档，可能会重复
            question = generate_single_question(llm, doc)
            guiding_questions.append(question)
    else:
        # 遍历每个文档组并生成引导性问题
        for source, docs in doc_groups.items():
            # 随机抽取部分内容生成问题
            selected_docs = random.sample(docs, min(num_questions_per_doc, len(docs)))
            
            for doc in selected_docs:
                question = generate_single_question(llm, doc)
                guiding_questions.append(question)
    
    # 如果生成的问题不足所需数量，随机选择文档继续生成
    while len(guiding_questions) < num_questions_total:
        random_source = random.choice(list(doc_groups.keys()))
        random_doc = random.choice(doc_groups[random_source])
        question = generate_single_question(llm, random_doc)
        guiding_questions.append(question)
    
    return guiding_questions[:num_questions_total]

def generate_single_question(llm, doc):
    variations = [
        "请根据以下内容提出一个引导性问题,不超过15个字。",
        "根据以下内容，生成一个引导性问题,不超过15个字。",
        "阅读以下内容后，提出一个引导性问题,不超过15个字。",
        "基于给定的文本，创建一个引导性问题,不超过15个字。",
        "考虑以下信息，形成一个引导性问题,不超过15个字。"
    ]
    random_variation = random.choice(variations)
    prompt = f"{random_variation}\n内容: {doc.page_content}\n，引导性问题内容要确保有主体，内容不能太复杂，问题要有引导意义，引导用户提问。只能生成一个问题不能有子问题，一步步思考，直接返回问题，不需要任何开头或解释。例如：深圳北站哪个出口最适合打滴滴？"
    
    result = llm.invoke(prompt).content
    preview_doc = _truncate_text(doc.page_content, 600)
    preview_prompt = f"{random_variation}\n内容: {preview_doc}\n，引导性问题内容要确保有主体，内容不能太复杂，问题要有引导意义，引导用户提问。只能生成一个问题不能有子问题，一步步思考，直接返回问题，不需要任何开头或解释。例如：深圳北站哪个出口最适合打滴滴？"
    _log_prompt(preview_prompt)
    
    return result.strip()

def get_document_snippets(documents, max_length=800):
    # 处理单个文档或文档列表
    if not isinstance(documents, list):
        documents = [documents]
    snippets = [doc.page_content[:max_length] for doc in documents]
    return "\n".join(snippets)

def document_question_relevance(question, documents):
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0,
        streaming=True
    )
    logger.debug(f"LLM model: {config['models']['llm_model']}")

    prompt = PromptTemplate(
        template="""你是一个评分员，评估检索到的文档与用户问题的相关性。
                    这里是检索到的文档:\n\n {document} \n\n
                    这是用户的问题:{question} \n
                    如果文档包含与用户问题相关的关键字，则将其评为相关。
                    它不需要是一个严格的测试。目标是过滤掉错误的检索。
                    给出一个二元分数"是"或"否"，以表明该文档是否与问题相关。
                    以JSON的形式提供二进制分数，其中只有一个关键字'score'，不需要任何开头或解释。""",
        input_variables=["question", "document"],
    )
    
    document = get_document_snippets(documents)
    retrieval_grader = prompt | llm | JsonOutputParser()
    
    result = retrieval_grader.invoke({"question": question, "document": document})
    return result['score']

def question_generation_from_last_dialogual(last_dialog):
    """根据上一轮对话生成引导性问题"""
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0,
        streaming=True
    )
    logger.debug(f"LLM model: {config['models']['llm_model']}")

    prompt = PromptTemplate(
        template="""你是一名问题引导专家，请根据上一轮的对话，提出3个引导性问题。每个问题之间用换行符分隔。
                    请注意：
                    问题前不需要任何序号。
                    问题内容要简洁明确，确保有明确的主题。
                    引导性问题要能够启发用户进一步思考或提问。
                    这是上一轮对话的记录：{last_dialog}
                    请直接生成问题，不需要任何开头或解释。请根据上述要求生成引导性问题：""",
        input_variables=["last_dialog"]
    )
    
    rag_chain = (
        {"last_dialog": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    derived_questions_str = rag_chain.invoke(last_dialog)
    derived_questions = [question for question in derived_questions_str.strip().split("\n") if question]
    return derived_questions

def run_llm_MulitDocQA(input_query: str, only_chatKBQA: bool, prompt_template_from_user: str, temperature: float, multiple_dialogue: bool, derivation: bool, show_source: bool, req_id=None):
    # 多文档 KB QA 主流程：
    # - 支持多轮对话：从 messages 中构造 history_str
    # - 检索改写：将当前问题改写为适合检索的独立问句
    # - 分支策略：
    #   * only_chatKBQA=True 严格依据 KB 回答
    #   * only_chatKBQA=False 检索优先，低相关时回退自由聊天
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=temperature,
        streaming=True
    )
    _pref = f"[req:{req_id}] " if req_id else ""
    logger.debug(f"{_pref}LLM model: {config['models']['llm_model']}")
    # global kb_state
    logger.debug(f"{_pref}Processing KB QA request")
    logger.debug(f"{_pref}KB loaded: {kb_state.current_kb_name}")
    
    # 处理多轮对话：从 messages 列表中提取问题和历史
    if multiple_dialogue and len(input_query) > 1:
        query = input_query[-1].content
        # 构建历史对话字符串
        history_items = []
        for msg in input_query[:-1]:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', '')
            if role == 'user':
                history_items.append(f"用户：{content}")
            elif role == 'assistant':
                history_items.append(f"AI：{content}")
        history_str = "\n".join(history_items) + "\n这是以上我和你的对话记录，请参考\n" if history_items else ""
    else:
        query = input_query[-1].content if input_query else ""
        history_str = ""
    
    logger.info(f"{_pref}Query: {query[:100]}..." if len(str(query)) > 100 else f"{_pref}Query: {query}")
    logger.debug(f"{_pref}Message count: {len(input_query)}, Multiple dialogue: {multiple_dialogue}")
    
    
    def generate_prompt(template, input_variables):
        return PromptTemplate(template=template, input_variables=input_variables)
    
    def create_chain(prompt):
        return prompt | llm | StrOutputParser()
    
    if only_chatKBQA:
        _retrieval_query = query
        if multiple_dialogue:
            try:
                llm_rewrite = ChatOpenAI(
                    api_key=config['paths']['openai_api_keys'],
                    base_url=config['paths']['openai_api_base'],
                    model=config['models']['llm_model'],
                    temperature=0.2,
                    streaming=False
                )
                rewrite_prompt = PromptTemplate(
                    template="请根据历史对话将当前问题改写为一个独立、明确的检索查询：\n历史：{history}\n当前问题：{query}\n改写后的检索查询：",
                    input_variables=["history", "query"]
                )
                # 检索改写：在多轮语境下生成更稳健的检索查询
                _retrieval_query = (rewrite_prompt | llm_rewrite | StrOutputParser()).invoke({"history": history_str, "query": query}).strip()
            except Exception:
                _retrieval_query = query
        top_documents_with_socre = get_top_documents(_retrieval_query, req_id=req_id)
        top_documents = [doc for doc, score in top_documents_with_socre]
        try:
            _names = [os.path.basename(doc.metadata.get('file_path', '')) for doc, _ in top_documents_with_socre if doc.metadata.get('file_path')]
            _scores = [score for _, score in top_documents_with_socre]
            logger.info(f"{_pref}selected_docs={_names} scores={_scores}")
        except Exception:
            pass

        # 阈值：
        # - rerank_direct_answer_threshold：直接使用文档内 QA 的最低分
        # - rerank_min_relevance：认为"与文档相关"的最低分（严格 KB 模式）
        thr = float(config['settings'].get('rerank_direct_answer_threshold', 0.8))
        thr_any = float(config['settings'].get('rerank_min_relevance', 0.2))
        score0 = (top_documents_with_socre[0][1] if top_documents_with_socre else 0.0)
        
        # 调试：显示得分和阈值
        logger.info(f"{_pref}重排得分: {score0:.4f}, 阈值: {thr_any}")
        
        # 特殊处理：对于Excel文档，如果检索到了但得分较低，可能是重排模型对表格格式理解不好
        # 检查是否所有文档都是Excel文件，如果是，可以适当降低阈值
        all_excel = all(doc.metadata.get('file_path', '').endswith(('.xlsx', '.xls', '.xlsm', '.csv')) 
                       for doc in top_documents if doc.metadata.get('file_path'))
        if all_excel and score0 > 0 and score0 < thr_any:
            # Excel文档的重排得分可能偏低，使用更宽松的阈值
            excel_threshold = max(0.01, thr_any * 0.1)  # 降低到原来的10%，但至少0.01
            logger.info(f"{_pref}检测到Excel文档，使用更宽松的阈值: {excel_threshold}")
            if score0 >= excel_threshold:
                logger.info(f"{_pref}Excel文档得分 {score0:.4f} 通过宽松阈值 {excel_threshold}")
                # 继续处理，不返回"未检索到"
                score0 = thr_any  # 临时提升得分，让它通过阈值检查
            else:
                logger.warning(f"{_pref}Excel文档得分 {score0:.4f} 仍低于宽松阈值 {excel_threshold}")
        
        direct = bool(top_documents) and bool(top_documents[0].metadata.get("isQA")) and score0 >= thr

        if direct:
            answer = "\n".join(top_documents[0].page_content.split('\n')[1:])
            logger.info(f"{_pref}direct_answer score={round(score0,2)}")
            if top_documents[0].metadata.get("file_url") and top_documents[0].metadata["file_url"] != '-':
                str_l = len(answer[:-1])
                try:
                    for i in range(str_l // 3):
                        logger.debug(answer[:-1][i*3:(i+1)*3])
                        yield stream_type(answer[:-1][i*3:(i+1)*3])
                        time.sleep(0.2)
                    yield stream_type(answer[:-1][(i+1)*3:(i+1)*3 + str_l%3])
                    time.sleep(0.2)
                    yield stream_type_url(answer[-1], top_documents[0].metadata["file_url"])
                except:
                    yield stream_type_url(answer, top_documents[0].metadata["file_url"])
            else:
                str_l = len(answer)
                try:
                    for i in range(str_l // 3):
                        logger.debug(answer[i*3:(i+1)*3])
                        yield stream_type(answer[i*3:(i+1)*3])
                        time.sleep(0.2)
                    yield stream_type(answer[(i+1)*3:(i+1)*3 + str_l%3])
                except:
                    yield stream_type(answer)
        else:
            if score0 < thr_any:
                # 严格 KB 模式下未命中：提示未检索到相关上下文
                yield f"data: {json.dumps(create_response_dict(content='没有检索到与查询相关的上下文信息,对不起,知识库中没有找到可以回答此问题的相关信息。', image_list=None, documents=None, sources=None), ensure_ascii=False)}\n\n".encode('utf-8')
            else:
                logger.info("文档和问题相关")
                # 强化上下文控制：防止超长报错 (400)
                max_total_chars = 12000
                max_doc_chars = 4000
                doc_texts = []
                for doc in top_documents:
                    content = _truncate_text(doc.page_content, max_doc_chars)
                    doc_texts.append(f"文件来源: {os.path.basename(doc.metadata.get('file_path', ''))}\n内容: {content}")
                
                # 合并并再次限长
                context_str = _truncate_text("\n\n---\n\n".join(doc_texts), max_total_chars)
                
                template = (prompt_template_from_user or "您是一位大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                            (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                            "\n以下是相关段落:{top_documents},下面是用户问题：{query} 回答："
                input_variables = ["query", "top_documents"]
                if multiple_dialogue:
                    input_variables.append("history")
                # 构造带有历史与相关段落的提示词
                prompt = generate_prompt(template, input_variables)
                chain = create_chain(prompt)
                inputs = {"query": query, "top_documents": context_str}
                if multiple_dialogue:
                    inputs["history"] = _truncate_text(history_str, 4000)
                    
                format_inputs = {k: inputs[k] for k in prompt.input_variables if k in inputs}
                rendered_safe = _render_prompt_safe(prompt, **format_inputs)
                _log_prompt(rendered_safe)
                
                # 记录核心提示词长度
                logger.debug(f"[req:{req_id}] 最终发送 Prompt 大致长度: {len(rendered_safe)}")
                
                response_text = ""
                for chunk in chain.stream(inputs):
                    response_text += chunk
                    yield stream_type(chunk)
    else:
        _retrieval_query = query
        if multiple_dialogue:
            try:
                llm_rewrite = ChatOpenAI(
                    api_key=config['paths']['openai_api_keys'],
                    base_url=config['paths']['openai_api_base'],
                    model=config['models']['llm_model'],
                    temperature=0.2,
                    streaming=False
                )
                rewrite_prompt = PromptTemplate(
                    template="请根据历史对话将当前问题改写为一个独立、明确的检索查询：\n历史：{history}\n当前问题：{query}\n改写后的检索查询：",
                    input_variables=["history", "query"]
                )
                # 检索改写：多轮对话下提炼独立问句
                _retrieval_query = (rewrite_prompt | llm_rewrite | StrOutputParser()).invoke({"history": history_str, "query": query}).strip()
            except Exception:
                _retrieval_query = query
        top_documents_with_socre = get_top_documents(_retrieval_query, req_id=req_id)
        # 自由聊天模式的回退阈值（可通过 config.settings.rerank_min_relevance_chat 调整）
        thr_any_chat = float(config['settings'].get('rerank_min_relevance_chat', 0.5))
        top_documents = [doc for doc, score in top_documents_with_socre]
        thr = float(config['settings'].get('rerank_direct_answer_threshold', 0.8))
        thr_any = float(config['settings'].get('rerank_min_relevance', 0.2))
        score0 = (top_documents_with_socre[0][1] if top_documents_with_socre else 0.0)
        direct = bool(top_documents) and bool(top_documents[0].metadata.get("isQA")) and score0 >= thr

        if direct:
            answer = "\n".join(top_documents[0].page_content.split('\n')[1:])
            logger.info(f"{_pref}direct_answer score={round(score0,2)}")
            if top_documents[0].metadata.get("file_url") and top_documents[0].metadata["file_url"] != '-':
                str_l = len(answer[:-1])
                try:
                    for i in range(str_l // 3):
                        logger.debug(answer[:-1][i*3:(i+1)*3])
                        yield stream_type(answer[:-1][i*3:(i+1)*3])
                        time.sleep(0.2)
                    yield stream_type(answer[:-1][(i+1)*3:(i+1)*3 + str_l%3])
                    time.sleep(0.2)
                    yield stream_type_url(answer[-1], top_documents[0].metadata["file_url"])
                except:
                    yield stream_type_url(answer, top_documents[0].metadata["file_url"])
            else:
                str_l = len(answer)
                try:
                    for i in range(str_l // 3):
                        logger.debug(answer[i*3:(i+1)*3])
                        yield stream_type(answer[i*3:(i+1)*3])
                        time.sleep(0.2)
                    yield stream_type(answer[(i+1)*3:(i+1)*3 + str_l%3])
                except:
                    yield stream_type(answer)
        else:
            if score0 < thr_any_chat:
                logger.debug("No relevant documents found, falling back to only_llm")
                for chunk in only_llm(input_query, prompt_template_from_user, temperature, multiple_dialogue):
                    yield chunk
            else:
                logger.info("文档和问题相关")
                # 强化上下文控制：防止超长报错 (400)
                max_total_chars = 12000
                max_doc_chars = 4000
                doc_texts = []
                for doc in top_documents:
                    content = _truncate_text(doc.page_content, max_doc_chars)
                    doc_texts.append(f"文件来源: {os.path.basename(doc.metadata.get('file_path', ''))}\n内容: {content}")
                
                context_str = _truncate_text("\n\n---\n\n".join(doc_texts), max_total_chars)
                
                template = (prompt_template_from_user or "您是一位大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                            (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                            "\n以下是相关段落:{top_documents},下面是用户问题：{query} 回答："
                input_variables = ["query", "top_documents"]
                if multiple_dialogue:
                    input_variables.append("history")
                prompt = generate_prompt(template, input_variables)
                chain = create_chain(prompt)
                inputs = {"query": query, "top_documents": context_str}
                if multiple_dialogue:
                    inputs["history"] = _truncate_text(history_str, 4000)
                format_inputs = {k: inputs[k] for k in prompt.input_variables if k in inputs}
                rendered_safe = _render_prompt_safe(prompt, **format_inputs)
                _log_prompt(rendered_safe)
                
                logger.debug(f"[req:{req_id}] 最终发送 Prompt 大致长度: {len(rendered_safe)}")
                
                response_text = ""
                for chunk in chain.stream(inputs):
                    response_text += chunk
                    yield stream_type(chunk)


def only_llm(input_query: str, prompt_template_from_user: str = "", temperature: float = 0.5, multiple_dialogue: bool = False):
    # 纯 LLM 对话：不依赖知识库检索；支持多轮，将 messages 转为历史注入提示词
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=temperature,
        streaming=True
    )
    logger.debug(f"LLM model: {config['models']['llm_model']}")
    
    # 处理多轮对话：从 messages 列表中提取问题和历史
    if multiple_dialogue and len(input_query) > 1:
        query = input_query[-1].content
        # 构建历史对话字符串
        history_items = []
        for msg in input_query[:-1]:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', '')
            if role == 'user':
                history_items.append(f"用户：{content}")
            elif role == 'assistant':
                history_items.append(f"AI：{content}")
        history_str = "\n".join(history_items) + "\n这是以上我和你的对话记录，请参考\n" if history_items else ""
    else:
        query = input_query[-1].content if input_query else ""
        history_str = ""
#    history_str = history_list_to_str(history)
    logger.debug(f"Query: {query[:100]}..." if len(str(query)) > 100 else f"Query: {query}")
    logger.debug(f"Message count: {len(input_query)}, Multiple dialogue: {multiple_dialogue}")
    
    
    def generate_prompt(template, input_variables):
        return PromptTemplate(template=template, input_variables=input_variables)
    
    def create_chain(prompt):
        return prompt | llm | StrOutputParser()
    
    # 通用对话提示词；如用户提供 system 提示，则以用户提示为准
    template = (prompt_template_from_user or "你是一位友好、专业的中文对话助手。请用清晰、自然的中文直接回答用户问题；在没有提供文档时进行自由对话与创作；如需讲故事或科普，请自行组织内容；保持简洁准确，不要声明无法回答。") + \
                    (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                    "\n下面是用户问题：{query} 回答："
                
    input_variables = ["query"]
    if multiple_dialogue:
        input_variables.append("history")
                
    prompt = generate_prompt(template, input_variables)
    chain = create_chain(prompt)
                
    inputs = {"query": query}
    if multiple_dialogue:
        inputs["history"] = _truncate_text(history_str, 4000)
    
    format_inputs = {k: inputs[k] for k in prompt.input_variables if k in inputs}
    rendered_safe = _render_prompt_safe(prompt, **format_inputs)
    _log_prompt(rendered_safe)
    response_text = ""
    for chunk in chain.stream(inputs):
        response_text += chunk
        yield stream_type(chunk)

def view_history(history):
    enc = tiktoken.get_encoding("cl100k_base")
    history_str = ""
    total_tokens = 0
    for item in history:
        query = item["query"]
        response = item["response"]
        history_str += f"User: {query}\nAssistant: {response}\n\n"
        total_tokens += len(enc.encode(query)) + len(enc.encode(response))
    return history_str, total_tokens

def get_uploaded_files():
    # Use kb_state.kb_vectordb (global state) instead of undefined kb_vectordb
    if getattr(kb_state, 'kb_vectordb', None) is None:
        return set()
    return {os.path.basename(doc.metadata.get('file_path', '')) for doc in kb_state.kb_vectordb.docstore._dict.values()}

def stream_type(data, model=config['models']['llm_model']):
    # SSE 输出格式：兼容前端增量渲染（choices[0].delta.content）
    return f"data: {json.dumps({'id': str(uuid.uuid4()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': data}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n".encode('utf-8')

def stream_type_url(data,file_url, model=config['models']['llm_model']):
    # SSE 输出格式（带来源链接）：用于直接答案附带文档 URL 的场景
    return f"data: {json.dumps({'id': str(uuid.uuid4()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': data,'file_url': file_url}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n".encode('utf-8')

def create_response_dict(**kwargs):
    return {
        "id": str(uuid.uuid4()),
        "model": config['models']['llm_model'],
        "choices": [{"index": 0, "delta": kwargs, "finish_reason": None}]
    }
#生成最后一个流式回复
def create_final_response(current_dialog, show_source, top_documents_with_score, derivation=False):
    # 最终响应（非流式聚合）构造：支持来源文档、派生问题与图片链接
    derived_questions = None

    # 如果 derivation 为 True，获取派生问题
    if derivation:
        derived_questions = question_generation_from_last_dialogual(str(current_dialog))

    # 准备最后一个包含所有信息的响应
    final_response = {
        "content": " ",
        "image_list": None,
        "doc": [],
        "derived_questions": derived_questions,
        "user_url": None
    }

    if show_source:
        # 创建一个字典来存储 source 到 document 列表的映射
        doc_dict = OrderedDict()
        for doc, score in top_documents_with_score:
            document_content = doc.page_content
            source = os.path.basename(doc.metadata.get('file_path', 'unknown'))
            
            if source not in doc_dict:
                doc_dict[source] = []
            
            doc_dict[source].append({
                "text": document_content,
                "mate": score
            })
        
        # 将字典转换为所需的 doc 列表格式
        for source, documents in doc_dict.items():
            final_response["doc"].append({
                "source": source, 
                "document": documents
            })
    top_documents = [doc for doc, score in top_documents_with_score]
    # 查找图片信息
    # 提取文档中的图片链接（Markdown 语法），用于前端展示
    image_info = find_image_links(top_documents)
    if image_info:
        final_response["image_list"] = image_info

    try:
        if top_documents[0].metadata["file_url"] != '-':
            final_response["user_url"] = top_documents[0].metadata["file_url"]
    except:
        pass

    return final_response
