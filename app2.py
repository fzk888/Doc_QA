from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from bm25_search import BM25Search
from typing import List,Dict
from Knowledge_based_async import KnowledgeBase
import asyncio
import os
import shutil
import yaml
import logging
import aiofiles
import json
from functions import (
    #load_vectordb_and_files,
    run_llm_Knowlege_baes_file_QA,
    run_llm_MulitDocQA,
    view_history,
    generate_guiding_questions,
    stream_type,
    embeddings,
    kb_state,
    only_llm
)
from functions import get_top_documents,create_final_response
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize FastAPI app
app = FastAPI()

# Global variables
# UPLOAD_DIRECTORY = "uploads" 
# os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
# #定义知识库状态


# response_text = ""

class Message(BaseModel):
    role: str
    content: str

class PromptRequest(BaseModel):
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
    current_dialog: Dict[str, str]
    show_source: bool
    derivation: bool
    query: str

# @app.on_event("startup")
# async def startup_event():
#     load_vectordb_and_files()


@app.post("/delete_kb")
async def delete_kb(kb_name: str = Form(...)):
    KB_dir = config['paths']['kb_dir']
    kb_dir = os.path.join(KB_dir, kb_name)

    if os.path.exists(kb_dir):
        try:
            shutil.rmtree(kb_dir)
            logger.info(f"Knowledge base '{kb_name}' deleted successfully")
            return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' deleted successfully"})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while deleting knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        logger.warning(f"Knowledge base '{kb_name}' does not exist")
        return JSONResponse(status_code=500, content={"code":500, "message": f"Knowledge base '{kb_name}' does not exist"})

@app.post("/update_vectordb")
async def update_vectordb_api(kb_name: str = Form(...), files: List[UploadFile] = File(...), state=kb_state):
    KB_DIR = config['paths']['kb_dir']
    kb_dir = os.path.join(KB_DIR, kb_name)
    upload_directory = os.path.join(kb_dir, "uploads")
    os.makedirs(upload_directory, exist_ok=True)

    kb = None

    # 检查知识库是否存在
    if not os.path.exists(kb_dir):
        try:
            os.makedirs(kb_dir, exist_ok=True)
            kb = KnowledgeBase(kb_name, embeddings)
            state.current_kb_name = kb_name
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        try:
            kb = KnowledgeBase(kb_name, embeddings)
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while loading knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

    # 保存上传的文件
    try:
        for file in files:
            file_path = os.path.join(upload_directory, file.filename)
            async with aiofiles.open(file_path, "wb") as buffer:
                await buffer.write(await file.read())
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while saving files for knowledge base '{kb_name}': {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

    try:
        files = [os.path.join(upload_directory, filename) for filename in os.listdir(upload_directory)]
        # 更新向量库
        print("# 更新向量库\n\n")
        result = await kb.update_vectordb(files)
        
        # 重新加载向量库
        state.kb_vectordb = await kb.load_vectordb()
        print('# 重新加载向量库\n\n')
        
        # 更新全局状态
        await update_global_state(kb_name, kb, state)
        print('# 更新全局状态\n\n')

        # 清空上传目录
        await asyncio.to_thread(shutil.rmtree, upload_directory)
        print('# 清空上传目录\n\n')
        
        os.makedirs(upload_directory)

        logger.info(f"Knowledge base '{kb_name}' updated successfully")
        
        return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' updated successfully and select", "result": result})

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while updating knowledge base '{kb_name}': {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

async def update_global_state(kb_name, kb, state):
    state.kb = kb
    state.kb_vectordb = await state.kb.load_vectordb()
    state.history = []
    state.unfilter_context = [doc for doc_id, doc in state.kb_vectordb.docstore._dict.items()]
    state.searcher_from_target_doc = BM25Search(state.unfilter_context)
    state.current_kb_name = kb_name
    print("重新选择向量库完成")
    logger.info(f"Knowledge base '{kb_name}' selected successfully")
    
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

#         logger.info(f"Knowledge base '{kb_name}' updated successfully")
        
#         # global kb_state
#         kb_state.kb = KnowledgeBase(kb_name, embeddings)
#         kb_state.kb_vectordb = kb_state.kb.load_vectordb()
#         kb_state.history = []
#         kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
#         kb_state.searcher_from_target_doc = BM25Search(kb_state.unfilter_context)
#         kb_state.current_kb_name = kb_name  # 更新当前知识库名称
#         print("重新选择向量库完成")
#         logger.info(f"Knowledge base '{kb_name}' selected successfully")

        
#         return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' updated successfully and select", "result": result})

    
#     except Exception as e:
#         error_message = str(e)
#         logger.error(f"Error occurred while updating knowledge base '{kb_name}': {error_message}")
#         return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

@app.post("/view_guiding_questions")
async def view_guiding_questions_api(request: Request):
    request_body = await request.json()
    kb_name = request_body.get('kb_name')

    #add 数据库加载
    KB_DIR = config['paths']['kb_dir']
    
    # global kb_state  # 确保使用全局的kb_state实例
    # 检查是否需要重新加载知识库
    if kb_state.current_kb_name == kb_name:
        logger.info(f"Knowledge base '{kb_name}' is already loaded")
    else:
        kb_dir = os.path.join(KB_DIR, kb_name)
        if os.path.exists(kb_dir):
            try:
                kb_state.kb = KnowledgeBase(kb_name, embeddings)
                kb_state.kb_vectordb = kb_state.kb.load_vectordb()
                kb_state.history = []
                kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
                kb_state.searcher_from_target_doc = BM25Search(kb_state.unfilter_context)
                kb_state.current_kb_name = kb_name  # 更新当前知识库名称
                logger.info(f"Knowledge base '{kb_name}' selected successfully")
                #yield JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' selected successfully"})
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error occurred while selecting knowledge base '{kb_name}': {error_message}")
                return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
        else:
            try:
                os.makedirs(kb_dir, exist_ok=True)
                kb_state.kb = KnowledgeBase(kb_name, embeddings)
                kb_state.kb_vectordb = None
                kb_state.history = []
                kb_state.unfilter_context = []
                kb_state.searcher_from_target_doc = None
                kb_state.current_kb_name = kb_name  # 更新当前知识库名称
                logger.info(f"Knowledge base '{kb_name}' created successfully")
                #return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' created successfully"})
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
                return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
#    except Exception as e:
#        error_message = str(e)
#        logger.error(f"An unexpected error occurred: {error_message}")
#        raise HTTPException(status_code=500, detail={"code": 500, "message":f"An unexpected error occurred: {error_message}"})
###############################################################
    try:
        guiding_questions = generate_guiding_questions()
        logger.info("Guiding questions generated successfully")
        return JSONResponse(status_code=200, content={"code": 200, "guiding_questions": guiding_questions})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while generating guiding questions: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

@app.post("/remove_file")
async def remove_file_api(kb_name: str = Form(...), file_name: str = Form(...)):
    KB_DIR = config['paths']['kb_dir']
    kb_dir = os.path.join(KB_DIR, kb_name)
    
    if not os.path.exists(kb_dir):
        logger.error(f"Knowledge base '{kb_name}' does not exist")
        raise HTTPException(status_code=500, detail={"code": 500,"message": f"Knowledge base '{kb_name}' does not exist"})
    
    try:
        kb = KnowledgeBase(kb_name, KB_DIR)
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
    try:
        request_data = await request.json()
        prompt_request = PromptRequest(**request_data)
        #add 数据库名字
        kb_name = prompt_request.kb_name
        print("接收到的名字")
        print(kb_name)
        messages = prompt_request.messages
        temperature = prompt_request.temperature
        prompt_template_from_user = messages[0].content
        query = messages[1:]
        only_chatKBQA = prompt_request.only_chatKBQA
        multiple_dialogue = prompt_request.multiple_dialogue
        derivation = prompt_request.derivation
        show_source = prompt_request.show_source
        if kb_name != None:
            #add 数据库加载
            KB_DIR = config['paths']['kb_dir']
            
            # global kb_state  # 确保使用全局的kb_state实例
            # 检查是否需要重新加载知识库
            if kb_state.current_kb_name == kb_name:
                print("不需要重新加载")
                logger.info(f"Knowledge base '{kb_name}' is already loaded")
    
            else:
                kb_dir = os.path.join(KB_DIR, kb_name)
        
                if os.path.exists(kb_dir):
                    try:
                        kb_state.current_kb_name = kb_name  # 更新当前知识库名称
                        kb_state.kb = KnowledgeBase(kb_name, embeddings)
                        kb_state.kb_vectordb = await kb_state.kb.load_vectordb()
                        # kb_state.kb_vectordb = kb_state.kb.load_vectordb()
                        kb_state.history = []
                        kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
                        kb_state.searcher_from_target_doc = BM25Search(kb_state.unfilter_context)
                        
                        print("重新加载成功1")
                        logger.info(f"Knowledge base '{kb_name}' selected successfully")
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
                        kb_state.kb = KnowledgeBase(kb_name, embeddings)
                        kb_state.kb_vectordb = None
                        kb_state.history = []
                        kb_state.unfilter_context = []
                        kb_state.searcher_from_target_doc = None
                        
                        print("重新加载成功2")
                        logger.info(f"Knowledge base '{kb_name}' created successfully")
                        #return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' created successfully"})
                    except Exception as e:
                        error_message = str(e)
                        logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
                        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
            
            result_generator = run_llm_MulitDocQA(query, only_chatKBQA, prompt_template_from_user, temperature, multiple_dialogue, derivation, show_source)
        else:
            result_generator = only_llm(query, only_chatKBQA, prompt_template_from_user, temperature, multiple_dialogue, derivation, show_source)

        async def output_generator():
            yield stream_type(None)
            response_text = ""
            for chunk in result_generator:
                decoded_chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                response_text += decoded_chunk
                yield chunk
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        logger.info("Request processed successfully")
        return StreamingResponse(output_generator(), media_type="text/event-stream")

    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(status_code=500, detail={"code": 500, "message":f"An unexpected error occurred: {error_message}"})
    
@app.post("/final_response")
async def get_final_response(request: FinalResponseRequest):
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
    try:
        #image_name = image_url
        #image_path = f"/root/autodl-tmp/project_/KG-LLM-Doc/Document_test/Doc_QA/{image_name}"
        image_path = image_url
        image_name = None
        if not os.path.exists(image_path):
            logger.error(f"Image '{image_name}' does not exist at path '{image_path}'")
            raise HTTPException(status_code=500, detail={"code": 500, "message": f"Image '{image_name}' does not exist at path '{image_path}'"})

        logger.info(f"Image '{image_name}' retrieved successfully from path '{image_path}'")
        return FileResponse(image_path)
    except HTTPException as e:
        logger.error(f"HTTP exception occurred: {e.detail}")
        raise e
    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(status_code=500, detail={"code": 500, "message": f"An unexpected error occurred: {error_message}"})

class QueryRequest(BaseModel):
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
            raise HTTPException(status_code=500, detail={"code": 500, "message": f"An unexpected error occurred: {s}"})


def custom_openapi():
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
