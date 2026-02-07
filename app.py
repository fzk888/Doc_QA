from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from core.search_bm25 import BM25Search
from typing import List,Dict
from core.kb_manager import KnowledgeBase
import asyncio
import os
import shutil
import yaml
import logging
import uuid
import time
from logging.handlers import RotatingFileHandler
import aiofiles
import json
from dotenv import load_dotenv

# 尽早加载 .env，确保后续导入的模块能读到环境变量（与工作目录无关）
_env_path = os.getenv("DOTENV_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

from core.engine import (
    run_llm_Knowlege_baes_file_QA,
    run_llm_MulitDocQA,
    view_history,
    generate_guiding_questions,
    stream_type,
    kb_state,
    only_llm,
    get_embeddings
)
from core.engine import get_top_documents, create_final_response
# Configure logging (console + rotating file)
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

logger = logging.getLogger("docqa")
logger.setLevel(logging.INFO)
logger.propagate = False

# Avoid duplicate handlers when reloading in dev
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(file_handler)

if not any(h for h in logger.handlers if getattr(h, "_is_console", False)):
    console_handler = logging.StreamHandler()
    console_handler._is_console = True
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(console_handler)

# Reduce noisy third-party logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS so the local frontend (served from file system or another port) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许任意来源（本地 file:// 与不同端口）
    allow_credentials=False,  # 避免浏览器在有凭据时拒绝 * 的 CORS 响应
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Simple request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    请求日志中间件，记录每个请求的基本信息和处理时间
    """
    start = time.time()
    method = request.method
    path = request.url.path
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    request.state.req_id = req_id
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info(f"[req:{req_id}] {method} {path} -> {response.status_code} in {duration_ms:.1f}ms")
        return response
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        logger.exception(f"[req:{req_id}] {method} {path} failed after {duration_ms:.1f}ms: {e}")
        raise

class Message(BaseModel):
    """
    消息模型，用于表示对话中的单条消息
    """
    role: str
    content: str

class PromptRequest(BaseModel):
    """
    提示请求模型，定义了API请求的数据结构
    """
    model: str = Field(default="Qwen1.5-32B-Chat")
    messages: List[Message]
    temperature: float = Field(default=0.5)
    n: int = Field(default=1)
    stream: bool = Field(default=True)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=4086)
    only_chatKBQA: bool = Field(default=True)
    keep_history: bool = Field(default=True)
    presence_penalty: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    repetition_penalty: float = Field(default=1.1)
    kb_name: str = Field(default=None)
    multiple_dialogue: bool = Field(default=False)
    derivation: bool = Field(default=False)
    show_source: bool = Field(default=False)

class FinalResponseRequest(BaseModel):
    """
    最终响应请求模型
    """
    current_dialog: Dict[str, str]
    show_source: bool
    derivation: bool
    query: str

# @app.on_event("startup")
# async def startup_event():
#     load_vectordb_and_files()


@app.get("/list_kb")
async def list_kb_api():
    """
    列出所有知识库
    
    Returns:
        JSONResponse: 包含知识库名称列表的响应
    """
    try:
        KB_DIR = config['paths']['kb_dir']
        if not os.path.exists(KB_DIR):
            return JSONResponse(status_code=200, content={"code": 200, "data": []})
        names = [name for name in os.listdir(KB_DIR) if os.path.isdir(os.path.join(KB_DIR, name))]
        logger.info(f"List KBs: {names}")
        return JSONResponse(status_code=200, content={"code": 200, "data": names})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Failed to list KBs: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

@app.post("/delete_kb")
async def delete_kb(kb_name: str = Form(...)):
    """
    删除指定的知识库
    
    Args:
        kb_name (str): 要删除的知识库名称
        
    Returns:
        JSONResponse: 删除结果响应
    """
    KB_dir = config['paths']['kb_dir']
    kb_dir = os.path.join(KB_dir, kb_name)

    if os.path.exists(kb_dir):
        try:
            shutil.rmtree(kb_dir)
            logger.info(f"✓ KB deleted: {kb_name}")
            return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' deleted successfully"})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while deleting knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        logger.warning(f"Knowledge base '{kb_name}' does not exist")
        return JSONResponse(status_code=500, content={"code":500, "message": f"Knowledge base '{kb_name}' does not exist"})

@app.post("/update_vectordb")
async def update_vectordb_api(kb_name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    更新向量数据库，处理上传的文件并构建知识库
    
    Args:
        kb_name (str): 知识库名称
        files (List[UploadFile]): 上传的文件列表
        
    Returns:
        JSONResponse: 更新结果响应
    """
    KB_DIR = config['paths']['kb_dir']
    kb_dir = os.path.join(KB_DIR, kb_name)
    upload_directory = os.path.join(kb_dir, "uploads")
    os.makedirs(upload_directory, exist_ok=True)

    kb = None
    # 使用全局 kb_state，避免 FastAPI/Pydantic 对默认参数进行深拷贝导致 _thread.lock 错误
    state = kb_state

    # 检查知识库是否存在
    if not os.path.exists(kb_dir):
        try:
            os.makedirs(kb_dir, exist_ok=True)
            kb = KnowledgeBase(kb_name, get_embeddings())
            state.current_kb_name = kb_name
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        try:
            kb = KnowledgeBase(kb_name, get_embeddings())
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while loading knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

    # 保存上传的文件，并保留副本到 doc_directory
    try:
        saved_paths = []
        for file in files:
            safe_name = os.path.basename(file.filename)
            file_path = os.path.join(upload_directory, safe_name)
            async with aiofiles.open(file_path, "wb") as buffer:
                content = await file.read()
                await buffer.write(content)
            # 保留一份到 doc_directory，便于用户核查原始文件
            try:
                doc_directory = os.path.join(kb.kb_dir, "doc_directory")
                os.makedirs(doc_directory, exist_ok=True)
                doc_copy_path = os.path.join(doc_directory, safe_name)
                await asyncio.to_thread(shutil.copy2, file_path, doc_copy_path)
            except Exception as copy_err:
                logger.warning(f"Copy to doc_directory failed for {safe_name}: {copy_err}")
            saved_paths.append(file_path)
        logger.info(f"Received {len(saved_paths)} file(s) for knowledge base '{kb_name}': {saved_paths}")
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while saving files for knowledge base '{kb_name}': {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

    try:
        files = [os.path.join(upload_directory, filename) for filename in os.listdir(upload_directory)]
        # 更新向量库
        print("# 更新向量库\n\n")
        logger.info(f"Start updating vector DB for KB '{kb_name}' with {len(files)} file(s)")
        result = await kb.update_vectordb(files)
        
        # 重新加载向量库
        state.kb_vectordb = await kb.load_vectordb()
        print('# 重新加载向量库\n\n')
        
        # 更新全局状态
        await update_global_state(kb_name, kb, state)
        print('# 更新全局状态\n\n')

        # 可选择：保留 uploads 目录中的文件。此处不再清空，便于用户确认上传成功
        # 如需节省空间，可改为移动到 doc_directory 后清空。
        logger.info(f"Knowledge base '{kb_name}' vector DB updated; uploads retained for verification.")

        logger.info(f"✓ KB updated: {kb_name}")
        
        return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' updated successfully and select", "result": result})

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while updating knowledge base '{kb_name}': {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

@app.get("/logs")
async def get_logs(lines: int = 200):
    """
    获取日志内容
    
    Args:
        lines (int): 返回的日志行数，默认200行
        
    Returns:
        JSONResponse: 包含日志内容的响应
    """
    try:
        if not os.path.exists(LOG_FILE):
            return JSONResponse(status_code=200, content={"code": 200, "message": "log file not found", "lines": []})
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        logs = content.splitlines()[-max(1, min(lines, 2000))]  # 防止一次性返回过多
        return JSONResponse(status_code=200, content={"code": 200, "lines": logs})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error reading logs: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

async def update_global_state(kb_name, kb, state):
    """
    更新全局状态，包括知识库实例、向量数据库、BM25搜索器等
    
    Args:
        kb_name (str): 知识库名称
        kb (KnowledgeBase): 知识库实例
        state: 全局状态对象
    """
    state.kb = kb
    state.kb_vectordb = await state.kb.load_vectordb()
    state.history = []
    # 防御性处理：向量库可能尚未创建或为空
    if state.kb_vectordb is not None:
        state.unfilter_context = [doc for doc_id, doc in state.kb_vectordb.docstore._dict.items()]
        logger.info("正在初始化 BM25 索引...")
        state.searcher_from_target_doc = BM25Search(state.unfilter_context)
    else:
        state.unfilter_context = []
        state.searcher_from_target_doc = None
    state.current_kb_name = kb_name
    print("重新选择向量库完成")
    logger.debug(f"KB selected: {kb_name}")
    
# @app.post("/update_vectordb")
# async def update_vectordb_api(kb_name: str = Form(...), files: List[UploadFile] = File(...), state=kb_state):
#     KB_DIR = config['paths']['kb_dir']
#     kb_dir = os.path.join(KB_DIR, kb_name)
#     upload_directory = os.path.join(kb_dir, "uploads")
#     os.makedirs(upload_directory, exist_ok=True)

#     kb = None  # 初始化知识库对象

#     # 检查知识库是否存在
#     if not os.path.exists(kb_dir):
#         try:
#             os.makedirs(kb_dir, exist_ok=True)
#             kb = KnowledgeBase(kb_name, embeddings)
#             state.current_kb_name = kb_name  # 更新当前知识库名称
#         except Exception as e:
#             error_message = str(e)
#             logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
#             return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
#     else:
#         try:
#             kb = KnowledgeBase(kb_name, embeddings)
#         except Exception as e:
#             error_message = str(e)
#             logger.error(f"Error occurred while loading knowledge base '{kb_name}': {error_message}")
#             return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

#     # 保存上传的文件
#     print(upload_directory)
#     #if file.filename in markdown_directory

    
#     try:
#         for file in files:
#             file_path = os.path.join(upload_directory, file.filename)
#             print("上传文件名",file)
#             async with aiofiles.open(file_path, "wb") as buffer:
#                 await buffer.write(await file.read())
#     except Exception as e:
#         error_message = str(e)
#         logger.error(f"Error occurred while saving files for knowledge base '{kb_name}': {error_message}")
#         return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

#     try:
#         files = [os.path.join(upload_directory, filename) for filename in os.listdir(upload_directory)]
#         # 更新向量库
#         result = kb.update_vectordb(files)
#         # 重新加载向量库
#         state.kb_vectordb = kb.load_vectordb()
#         # 清空历史记录
#         state.history = []
#         # 更新未过滤的上下文
#         state.unfilter_context = [doc for doc_id, doc in state.kb_vectordb.docstore._dict.items()]
#         # 创建新的搜索器
#         state.searcher_from_target_doc = BM25Search(state.unfilter_context)

#         # 清空上传目录
#         shutil.rmtree(upload_directory)
#         os.makedirs(upload_directory)

#         logger.info(f"✓ KB updated: {kb_name}")
        
#         # global kb_state
#         kb_state.kb = KnowledgeBase(kb_name, embeddings)
#         kb_state.kb_vectordb = kb_state.kb.load_vectordb()
#         kb_state.history = []
#         kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
#         kb_state.searcher_from_target_doc = BM25Search(kb_state.unfilter_context)
#         kb_state.current_kb_name = kb_name  # 更新当前知识库名称
#         print("重新选择向量库完成")
#         logger.debug(f"KB selected: {kb_name}")

        
#         return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' updated successfully and select", "result": result})

    
#     except Exception as e:
#         error_message = str(e)
#         logger.error(f"Error occurred while updating knowledge base '{kb_name}': {error_message}")
#         return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

@app.post("/view_guiding_questions")
async def view_guiding_questions_api(request: Request):
    """
    查看引导性问题
    
    Args:
        request (Request): HTTP请求对象
        
    Returns:
        JSONResponse: 包含引导性问题列表的响应
    """
    try:
        request_body = await request.json()
        kb_name = request_body.get('kb_name')

        #add 数据库加载
        KB_DIR = config['paths']['kb_dir']
        
        # global kb_state  # 确保使用全局的kb_state实例
        # 检查是否需要重新加载知识库
        if kb_state.current_kb_name == kb_name:
            logger.debug(f"KB {kb_name} cached")
        else:
            kb_dir = os.path.join(KB_DIR, kb_name)
            if os.path.exists(kb_dir):
                try:
                    kb_state.kb = KnowledgeBase(kb_name, get_embeddings())
                    kb_state.kb_vectordb = await kb_state.kb.load_vectordb()
                    kb_state.history = []
                    # 防御：向量库可能尚未创建
                    if kb_state.kb_vectordb is not None:
                        kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
                        kb_state.searcher_from_target_doc = BM25Search(kb_state.unfilter_context)
                    else:
                        kb_state.unfilter_context = []
                        kb_state.searcher_from_target_doc = None
                    kb_state.current_kb_name = kb_name  # 更新当前知识库名称
                    logger.debug(f"KB selected: {kb_name}")
                    #yield JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' selected successfully"})
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error occurred while selecting knowledge base '{kb_name}': {error_message}")
                    return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
            else:
                try:
                    os.makedirs(kb_dir, exist_ok=True)
                    kb_state.kb = KnowledgeBase(kb_name, get_embeddings())
                    kb_state.kb_vectordb = None
                    kb_state.history = []
                    kb_state.unfilter_context = []
                    kb_state.searcher_from_target_doc = None
                    kb_state.current_kb_name = kb_name  # 更新当前知识库名称
                    logger.info(f"✓ KB created: {kb_name}")
                    #return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' created successfully"})
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
                    return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(status_code=500, detail={"code": 500, "message":f"An unexpected error occurred: {error_message}"})
###############################################################
    try:
        # 若尚未构建向量库，直接返回空的引导问题以避免 500
        if kb_state.kb_vectordb is None:
            logger.warning(f"Vector DB not available for knowledge base '{kb_state.current_kb_name}'. Returning empty guiding questions.")
            return JSONResponse(status_code=200, content={"code": 200, "guiding_questions": []})

        guiding_questions = generate_guiding_questions()
        logger.info("Guiding questions generated successfully")
        return JSONResponse(status_code=200, content={"code": 200, "guiding_questions": guiding_questions})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while generating guiding questions: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

@app.post("/remove_file")
async def remove_file_api(kb_name: str = Form(...), file_name: str = Form(...)):
    """
    从知识库中删除指定文件
    
    Args:
        kb_name (str): 知识库名称
        file_name (str): 要删除的文件名
        
    Returns:
        JSONResponse: 删除结果响应
    """
    KB_DIR = config['paths']['kb_dir']
    kb_dir = os.path.join(KB_DIR, kb_name)
    
    if not os.path.exists(kb_dir):
        logger.error(f"Knowledge base '{kb_name}' does not exist")
        raise HTTPException(status_code=500, detail={"code": 500,"message": f"Knowledge base '{kb_name}' does not exist"})
    
    try:
        kb = KnowledgeBase(kb_name, get_embeddings())
        result = await kb.remove_file(file_name)
        logger.info(f"File '{file_name}' removed successfully from knowledge base '{kb_name}'")
        return JSONResponse(status_code=200, content={"code": 200, "message": result})
    except HTTPException as e:
        logger.error(f"HTTP exception occurred: {e.detail}")
        raise e
    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(status_code=500,detail={"code": 500, "message": "An unexpected error occurred"})

@app.post("/mulitdoc_qa")
async def run_llm_mulitdoc_qa_api(request: Request):
    """
    多文档问答接口
    
    Args:
        request (Request): HTTP请求对象，包含用户查询和相关参数
        
    Returns:
        StreamingResponse 或 JSONResponse: 流式响应或JSON响应
    """
    try:
        request_data = await request.json()
        prompt_request = PromptRequest(**request_data)
        #add 数据库名字
        kb_name = prompt_request.kb_name
        _req = getattr(request.state, "req_id", None) or uuid.uuid4().hex[:8]
        logger.info(f"[req:{_req}] /mulitdoc_qa received kb={kb_name}")
        messages = prompt_request.messages
        temperature = prompt_request.temperature
        # 正确处理消息内容：仅当存在 system 角色消息时才作为提示词模板
        prompt_template_from_user = ""
        for m in messages:
            try:
                if getattr(m, 'role', '') == 'system' and getattr(m, 'content', ''):
                    prompt_template_from_user = m.content
                    break
            except Exception:
                pass
        
        # 将整个messages列表传递给函数，而不是切片
        query = messages
        only_chatKBQA = prompt_request.only_chatKBQA
        multiple_dialogue = prompt_request.multiple_dialogue
        derivation = prompt_request.derivation
        show_source = prompt_request.show_source
        if 'only_chatKBQA' not in request_data:
            only_chatKBQA = config['settings'].get('only_chatKBQA_default', True)
        if 'temperature' not in request_data:
            temperature = config['settings'].get('temperature_default', 0.5)
        logger.info(f"[req:{_req}] flags only_chatKBQA={only_chatKBQA} multiple_dialogue={multiple_dialogue} derivation={derivation} show_source={show_source} stream={getattr(prompt_request,'stream',True)}")

        if kb_name != None:
            #add 数据库加载
            KB_DIR = config['paths']['kb_dir']
            
            # global kb_state  # 确保使用全局的kb_state实例
            # 检查是否需要重新加载知识库
            if kb_state.current_kb_name == kb_name:
                logger.debug(f"[req:{_req}] KB {kb_name} cached")
    
            else:
                kb_dir = os.path.join(KB_DIR, kb_name)
        
                if os.path.exists(kb_dir):
                    try:
                        kb_state.current_kb_name = kb_name
                        kb_state.kb = KnowledgeBase(kb_name, get_embeddings())
                        kb_state.kb_vectordb = await kb_state.kb.load_vectordb()
                        kb_state.history = []
                        # 防御：向量库可能尚未创建
                        if kb_state.kb_vectordb is not None:
                            kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
                            kb_state.searcher_from_target_doc = BM25Search(kb_state.unfilter_context)
                        else:
                            kb_state.unfilter_context = []
                            kb_state.searcher_from_target_doc = None
                            logger.warning(f"Vector DB not available for knowledge base '{kb_name}'. Proceeding without KB context.")
                        
                        logger.debug(f"[req:{_req}] KB selected: {kb_name}")
                        #yield JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' selected successfully"})
                    except Exception as e:
                        error_message = str(e)
                        logger.error(f"Error occurred while selecting knowledge base '{kb_name}': {error_message}")
                        #yield 
                        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
                else:
                    try:
                        kb_state.current_kb_name = kb_name  # 更新当前知识库名称
                        os.makedirs(kb_dir, exist_ok=True)
                        kb_state.kb = KnowledgeBase(kb_name, get_embeddings())
                        kb_state.kb_vectordb = None
                        kb_state.history = []
                        kb_state.unfilter_context = []
                        kb_state.searcher_from_target_doc = None
                        
                        logger.info(f"[req:{_req}] ✓ KB created: {kb_name}")
                        #return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' created successfully"})
                    except Exception as e:
                        error_message = str(e)
                        logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
                        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
            
            # 当向量库不可用时，回退为仅LLM对话，避免 500
            if kb_state.kb_vectordb is None:
                logger.warning(f"[req:{_req}] KB '{kb_name}' has no vector DB; answering without KB context.")
                result_generator = only_llm(query, prompt_template_from_user, temperature, multiple_dialogue)
            else:
                logger.info(f"[req:{_req}] using KB '{kb_name}' for QA")
                result_generator = run_llm_MulitDocQA(query, only_chatKBQA, prompt_template_from_user, temperature, multiple_dialogue, derivation, show_source, req_id=_req)
        else:
            logger.info(f"[req:{_req}] no kb specified; using only LLM")
            result_generator = only_llm(query, prompt_template_from_user, temperature, multiple_dialogue)

        # 如果客户端支持流式（SSE），按流式返回；否则聚合为一次性 JSON 返回
        stream = getattr(prompt_request, "stream", True)
        if stream:
            async def output_generator():
                yield stream_type(None)
                response_text = ""
                for chunk in result_generator:
                    decoded_chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                    response_text += decoded_chunk
                    yield chunk
                    await asyncio.sleep(0)
                yield "data: [DONE]\n\n"

            logger.info(f"[req:{_req}] Request processed successfully")
            # 添加禁用缓冲的响应头，确保反向代理和浏览器即时刷新 SSE 内容
            return StreamingResponse(
                output_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式聚合：解析 SSE payload 提取文本
            aggregated_text = ""
            sources = []
            for chunk in result_generator:
                decoded_chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
                if not decoded_chunk.startswith("data: "):
                    continue
                payload = decoded_chunk[len("data: "):].strip()
                try:
                    obj = json.loads(payload)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        aggregated_text += content
                    file_url = delta.get("file_url")
                    if show_source and file_url:
                        sources.append(file_url)
                except Exception:
                    # 跳过异常的 SSE 片段
                    continue

            logger.info(f"[req:{_req}] Request processed successfully (non-stream)")
            return JSONResponse(status_code=200, content={
                "code": 200,
                "message": "success",
                "data": {
                    "answer": aggregated_text,
                    "sources": sources if show_source else None
                }
            })

    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(status_code=500, detail={"code": 500, "message":f"An unexpected error occurred: {error_message}"})
    
@app.post("/final_response")
async def get_final_response(request: FinalResponseRequest):
    """
    获取最终响应，包括文档来源、派生问题等信息
    
    Args:
        request (FinalResponseRequest): 最终响应请求对象
        
    Returns:
        JSONResponse: 包含完整响应信息的JSON响应
    """
    top_documents_with_score = get_top_documents(request.query)
    
    final_response = create_final_response(
        request.current_dialog, 
        request.show_source, 
        top_documents_with_score, 
        request.derivation
    )
    
    return JSONResponse(content=final_response)

@app.get("/display_image")
async def display_image(image_url: str):
    """
    显示图片
    
    Args:
        image_url (str): 图片URL
        
    Returns:
        FileResponse: 图片文件响应
    """
    try:
        from urllib.parse import unquote, urlparse
        u = unquote(image_url.strip())
        parsed = urlparse(u)
        p = u
        if parsed.scheme == "file":
            p = parsed.path
        if p.startswith("/") and len(p) > 3 and p[1].isalpha() and p[2] == ":":
            p = p[1:]
        p = os.path.normpath(p.replace("/", os.sep))
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.getcwd(), p))
        if not os.path.exists(p):
            raise HTTPException(status_code=500, detail={"code": 500, "message": f"Image does not exist at path '{p}'"})
        return FileResponse(p)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": 500, "message": str(e)})

class QueryRequest(BaseModel):
    """
    查询请求模型
    """
    model: str = Field(default="Qwen1.5-32B-Chat")
    messages: List[Message]
    temperature: float = Field(default=0.5)
    n: int = Field(default=1)
    stream: bool = Field(default=True)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=4086)
    keep_history: bool = Field(default=True)
    presence_penalty: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    repetition_penalty: float = Field(default=1.1)

@app.post("/Knowlege_baes_file_QA")
async def run_llm_Knowlege_baes_file_QA_api(request: Request):
    """
    基于知识库文件的问答接口
    
    Args:
        request (Request): HTTP请求对象
        
    Returns:
        StreamingResponse: 流式响应
    """
    try:
        request_data = await request.json()
        prompt_request = PromptRequest(**request_data)
        messages = prompt_request.messages
        query = messages[0].content
        keep_history = prompt_request.keep_history

        result_generator = run_llm_Knowlege_baes_file_QA(query, keep_history)

        async def output_generator():
            yield stream_type(None)
            for chunk in result_generator:
                yield stream_type(chunk)
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(output_generator(), media_type="text/event-stream")

    except Exception as e:
        error_message = str(e)
        if "Content Exists Risk" in error_message:
            print("====")
            raise HTTPException(status_code=500, detail={"code": 500, "message": "Your content contains sensitive information. Please rephrase your question."})
        else:
            raise HTTPException(status_code=500, detail={"code": 500, "message": f"An unexpected error occurred: {error_message}"})


def custom_openapi():
    """
    自定义OpenAPI文档
    
    Returns:
        dict: OpenAPI文档字典
    """
    if app.openapi_schema:
        return app.openapi_schemas
    openapi_schema = get_openapi(
        title="知识库问答应用",
        version="1.0.0",
        description="一个基于知识库的问答应用",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=7861)